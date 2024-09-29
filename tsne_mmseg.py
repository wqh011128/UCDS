import os
from argparse import ArgumentParser
from tqdm import tqdm
import torch.nn.functional as F
import mmcv
from tools.test import update_legacy_cfg
import torch
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette
from mmseg.datasets.builder import build_dataset
from mmseg.datasets.builder import build_dataloader
from mmseg.models.builder import build_segmentor
from mmseg.core.evaluation import get_classes, get_palette
import numpy as np
from time import time
import os.path as osp
from openTSNE import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()
    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])

    test_dataset = build_dataset(cfg.data.test)
    test_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
# target
    features, labels = [], []
    with torch.no_grad():
        for i, data_dict in tqdm(enumerate(test_loader)):
            # 提取特征和标签
            x = data_dict['img'][0]
            y = data_dict['gt_semantic_seg'][0]

            # 获取模型的特征
            feature = model.encode_decode(img=x.cuda(), img_metas=data_dict['img_metas'][0])

            # 将特征展平为 [total_pixels, feature_dim]
            features_flat = feature.permute(0, 2, 3, 1).reshape(-1, 19).detach().cpu().numpy()  # 转换为numpy数组
            labels_flat = y.reshape(-1).detach().cpu().numpy()  # [total_pixels] 类别标签

            # 指定要可视化的类别
            desired_classes = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # 类别 255 可以忽略

            # 筛选出属于指定类别的像素
            mask = np.isin(labels_flat, desired_classes)  # 生成一个布尔掩码，表示哪些像素属于目标类别
            filtered_features = features_flat[mask]  # 过滤出对应的特征
            filtered_labels = labels_flat[mask]  # 过滤出对应的标签

            # 如果你还想随机采样5000个像素，可以继续采样
            num_samples = min(30000, filtered_features.shape[0])  # 确保样本数不超过总像素数
            sample_indices = np.random.choice(filtered_features.shape[0], num_samples, replace=False)

            sampled_features = filtered_features[sample_indices]
            sampled_labels = filtered_labels[sample_indices]

            # 使用openTSNE进行降维
            tsne = TSNE(n_components=2, perplexity=100, metric="cosine", n_jobs=4)
            features_tsne = tsne.fit(sampled_features)

            # 定义新的颜色映射，确保包含所有 desired_classes 对应的颜色
            cityscapes_colors = [
                (128, 64, 128),  # road
                (70, 70, 70),  # building (class 2)
                (102, 102, 156),  # wall (class 3)
                (190, 153, 153),  # fence (class 4)
                (153, 153, 153),  # pole (class 5)
                (250, 170, 30),  # traffic light (class 6)
                (220, 220, 0),  # traffic sign (class 7)
                (107, 142, 35),  # vegetation (class 8)
                (152, 251, 152),  # terrain (class 9)r
                (0, 130, 180),  # sky (class 10)
                (220, 20, 60),  # person (class 11)
                (255, 0, 0),  # rider (class 12)
                (0, 0, 142),  # car (class 13)
            ]
            cityscapes_colors = [(r / 255, g / 255, b / 255) for r, g, b in cityscapes_colors]
            cityscapes_cmap = ListedColormap(cityscapes_colors)

            # 可视化并根据类别标签上色
            plt.figure(figsize=(8, 8))
            plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=sampled_labels, cmap=cityscapes_cmap, s=5)
            plt.axis("off")
            # plt.colorbar()
            # plt.title("t-SNE Visualization for Specific Classes")
            plt.show()
if __name__ == '__main__':
    main()
