# tools/get_flops.py
import argparse
from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmseg.models import build_segmentor
import time
import os.path as osp
# tools/benchmark.py
import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs and fps of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512,512],
        help='input image size')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the results will be dumped '
              'into the directory as json'))
    parser.add_argument('--repeat-times', type=int, default=1)
    args = parser.parse_args()
    return args


def get_params_flops():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


def get_fps():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.work_dir is not None:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        json_file = osp.join(args.work_dir, f'fps_{timestamp}.json')
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        json_file = osp.join(work_dir, f'fps_{timestamp}.json')

    repeat_times = args.repeat_times
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    benchmark_dict = dict(config=args.config, unit='img / s')
    overall_fps_list = []
    for time_index in range(repeat_times):
        print(f'Run {time_index + 1}:')
        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if 'checkpoint' in args and osp.exists(args.checkpoint):
            load_checkpoint(model, args.checkpoint, map_location='cpu')

        model = MMDataParallel(model, device_ids=[0])

        model.eval()

        # the first several iterations may be very slow so skip them
        num_warmup = 5
        pure_inf_time = 0
        total_iters = 200

        # benchmark with 200 image and take the average
        for i, data in enumerate(data_loader):

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % args.log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Done image [{i + 1:<3}/ {total_iters}], '
                          f'fps: {fps:.2f} img / s')

            if (i + 1) == total_iters:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall fps: {fps:.2f} img / s\n')
                benchmark_dict[f'overall_fps_{time_index + 1}'] = round(fps, 2)
                overall_fps_list.append(fps)
                break
    benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
    benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 4)
    print(f'Average fps of {repeat_times} evaluations: '
          f'{benchmark_dict["average_fps"]}')
    print(f'The variance of {repeat_times} evaluations: '
          f'{benchmark_dict["fps_variance"]}')
    mmcv.dump(benchmark_dict, json_file, indent=4)


if __name__ == '__main__':
    get_params_flops()
    get_fps()