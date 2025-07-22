import argparse
import torch
import numpy as np
from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics.pairwise import rbf_kernel
import math
from tqdm import tqdm
from argparse import ArgumentParser
from tools.test import update_legacy_cfg
import torch.nn.functional as F
from mmseg.core.evaluation import get_classes, get_palette
from mmseg.apis import inference_segmentor, init_segmentor

import mmcv
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
    # 构建目标数据集和数据加载器
    target_dataset = build_dataset(cfg.data.test)
    target_loader = build_dataloader(
        target_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # 构建源数据集和数据加载器
    source_dataset = build_dataset(cfg.data.val)
    source_loader = build_dataloader(
        source_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # 提取特征
    print("Extracting features from source dataset...")
    source_features = extract_features(model, source_loader, args.device)
    print("Extracting features from target dataset...")
    target_features = extract_features(model, target_loader, args.device)

    # 组织特征按类别
    print("Organizing features by class...")
    source_class_features = organize_features_by_class(source_features, source_dataset.CLASSES)
    target_class_features = organize_features_by_class(target_features, target_dataset.CLASSES)

    # 计算并打印每个类别的MMD距离
    print("Computing MMD distances per class...")
    for cls in target_dataset.CLASSES:
        if cls in source_class_features and cls in target_class_features:
            mmd = compute_mmd(source_class_features[cls], target_class_features[cls])
            print(f"MMD distance for class '{cls}': {mmd:.4f}")
        else:
            print(f"Class '{cls}' not found in both datasets.")


def extract_features(model, data_loader, device):
    """
    使用模型提取特征。
    假设我们提取的是模型的某一层的输出，例如最后一个卷积层。
    根据您的模型架构，您可能需要调整提取特征的方法。
    """
    features = []
    with torch.no_grad():
        for i, data_dict in tqdm(enumerate(data_loader)):
            inp = data_dict['img'][0].cuda()
            x = model.extract_feat(inp)
            label = data_dict['gt_semantic_seg'][0]
            label = F.interpolate(label.unsqueeze(0), size=(x[-1].shape[-2], x[-1].shape[-1]), mode='nearest').squeeze(0)
            features.append({
                'features': x[-1][0].cpu().numpy(),
                'labels': label[0].cpu().numpy()
            })
            if i == 500:
                break
    return features


def organize_features_by_class(features, class_names):
    """
    将提取的特征按类别组织。
    假设每个特征对应一个像素点，并且有对应的类别标签。
    """
    class_features = defaultdict(list)
    for sample in features:
        feat = sample['features']  # 形状可能为 (C, H, W) 或其他
        labels = sample['labels']  # 形状为 (H, W)
        C = feat.shape[0]
        H = feat.shape[1]
        W = feat.shape[2]
        # 遍历每个像素，按类别收集特征
        for cls_idx, cls in enumerate(class_names):
            mask = labels == cls_idx
            if mask.sum() == 0:
                continue
            cls_feat = feat[:, mask]  # 形状为 (C, N)
            cls_feat = cls_feat.transpose(1, 0)  # 转换为 (N, C)
            class_features[cls].append(cls_feat)
    # 将每个类别的特征拼接起来
    for cls in class_features:
        class_features[cls] = np.concatenate(class_features[cls], axis=0)
    return class_features


def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    """
    计算两个样本集的MMD距离。
    x: numpy array of shape (n_samples_x, n_features)
    y: numpy array of shape (n_samples_y, n_features)
    """
    xx, yy, zz = np.matmul(x, x.T), np.matmul(y, y.T), np.matmul(x, y.T)
    if kernel == 'rbf':
        # 计算带宽参数
        gamma = 1.0 / (2 * sigma ** 2)
        K = rbf_kernel(x, x, gamma=gamma)
        L = rbf_kernel(y, y, gamma=gamma)
        P = rbf_kernel(x, y, gamma=gamma)
    else:
        raise NotImplementedError("仅支持RBF核")

    m = x.shape[0]
    n = y.shape[0]

    mmd = K.sum() / (m * m) + L.sum() / (n * n) - 2 * P.sum() / (m * n)
    return math.sqrt(mmd)


if __name__ == '__main__':
    main()
