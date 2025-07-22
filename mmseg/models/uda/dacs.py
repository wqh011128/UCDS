# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.utils.utils import downscale_label_ratio
import logging

import torch.nn as nn

from segment_anything import sam_model_registry, SamPredictor
from timm.models.layers import trunc_normal_
from mmseg.ops import resize

class Sam_vit_h(nn.Module):
    def __init__(self, model_type="vit_h", sam_checkpoint="/data/wuqihang/work2/sam_checkpoint/sam_vit_h_4b8939.pth"):
        super(Sam_vit_h, self).__init__()
        self.model_type = model_type
        self.sam_checpoint = sam_checkpoint
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checpoint)
        self.predictor = SamPredictor(self.sam)

    def forward_last_image_encoder_feature(self, input):
        features = self.predictor.model.image_encoder(input)
        return features

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm
class Sam_Consistency(nn.Module):
    def __init__(self, mask_ratio, embed_dim=3, block_size=64):
        super(Sam_Consistency, self).__init__()
        self.sam_model = Sam_vit_h(model_type="vit_b",
                                   sam_checkpoint="/data/wuqihang/work2/sam_checkpoint/sam_vit_b_01ec64.pth")  # 不需要指定设备，我们会将其嵌入到DACS类中一起scatter设别
        self.mask_ratio = mask_ratio  # 掩码率
        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)  # share mask_token without pos
        self.block_size = block_size  # 掩码块大小
        self.proj = nn.Sequential(  # proj到sam feature相同的空间，这里仍然是只有最后一层，如果用多层需要多个定义
            nn.Conv2d(
                in_channels=512,
                out_channels=(256 * 4 ** 2),
                kernel_size=1
            ),
            nn.PixelShuffle(4)
        )
        trunc_normal_(self.mask_token, mean=0., std=.02)
        self.loss_weight = 0.5

    def mask_input(self, input):
        B, _, H, W = input.shape
        mshape = B, 1, round(H / self.block_size), round(
            W / self.block_size)
        input_mask = torch.rand(mshape, device=input.device)
        input_mask = (input_mask > self.mask_ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        masked_img = input * input_mask + (1. - input_mask) * self.mask_token.expand(B, -1, H, W)

        # np_img = masked_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # plt.imshow(np_img)
        # plt.title('Random Masked Image')
        # plt.axis('off')
        # plt.show()
        return masked_img

    def forward(self, student_model, target_img):
        align_loss = dict()
        # 传入学生模型和目标域图像
        # 对学生模型的前向传播进行修改，加入掩码
        masked_img = self.mask_input(target_img)
        stu_features = student_model.extract_lr_mask_feature_only(masked_img)
        # 推理sam的encoder
        with torch.no_grad():
            sam_features = self.sam_model.forward_last_image_encoder_feature(target_img)
        # 投影stu
        stu_features = self.proj(stu_features[-1])
        # alignment
        align_loss["loss_last_level"] = self.loss_weight * F.mse_loss(stu_features, sam_features)
        return align_loss

def entropy(p, dim=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim)

def compute_pw(ema_softmax):
    B, C, H, W = ema_softmax.shape
    ema_softmax = ema_softmax.permute(0, 2, 3, 1).reshape(-1, C)
    w = entropy(ema_softmax)
    w = torch.exp(-1.0 * w)
    w = w.reshape(B, H, W)
    return w
def label_onehot(inputs, num_segments):
    if inputs.ndim == 4:
        inputs.squeeze(1)
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_segments, batch_size, im_h, im_w)).cuda()

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    outputs.scatter_(0, inputs_temp.unsqueeze(1), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)

@torch.no_grad()
def compute_neighbors(refine_ema_logits, refine_check_logits, bank, k=3, train_time=0):
    memory_bank = bank['memory_bank']
    refine_bank = bank['refine_bank']
    C = memory_bank[0].shape[-1]
    ema_logits = F.normalize(refine_ema_logits, p=2, dim=-1)
    check_logits = F.normalize(refine_check_logits, p=2, dim=-1)
    unnormalized_memory_bank = torch.stack(memory_bank, dim=0).reshape(-1, C)
    unnormalized_refine_bank = torch.stack(refine_bank, dim=0).reshape(-1, C)
    normalized_memory_bank = F.normalize(unnormalized_memory_bank, p=2, dim=-1)
    # 防止爆显存
    ema_logits = torch.chunk(ema_logits, 500, dim=0)
    check_logits = torch.chunk(check_logits, 500, dim=0)
    ema_indices, check_indices, log_indices = [], [], []
    for ema_logit, check_logit in zip(ema_logits, check_logits):
        ema_cos = torch.matmul(ema_logit, normalized_memory_bank.T)
        ema_topk_cos, ema_indice = torch.topk(ema_cos, k=k, dim=-1)
        ema_indices.append(ema_indice)
        log_indices.append(ema_topk_cos[:, -1] < 0.968 + pow(train_time, 0.2) * 0.032)
        check_cos = torch.matmul(check_logit, normalized_memory_bank.T)
        check_indice = torch.topk(check_cos, k=k, dim=-1)[1]
        check_indices.append(check_indice)
    ema_indices = torch.cat(ema_indices, dim=0)
    check_indices = torch.cat(check_indices, dim=0)
    log_indices = torch.cat(log_indices, dim=0)
    # N, 2k
    indices = torch.cat([ema_indices, check_indices], dim=-1)
    M, N = unnormalized_memory_bank.shape[0], indices.shape[0]
    # use unnormalized_bank to refine pseudo labels
    expand_bank = unnormalized_refine_bank.unsqueeze(0).expand(N, M, C)
    indices = indices.unsqueeze(-1).expand(N, 2 * k, C)
    output = F.softmax(torch.gather(expand_bank, 1, indices), dim=-1)
    # 加权-k=4
    # weight_matrix = torch.ones_like(output, device='cuda')
    # weight_matrix[:, 0, :], weight_matrix[:, 4, :] = 0.1, 0.1
    # weight_matrix[:, 1, :], weight_matrix[:, 5, :] = 0.05, 0.05
    # weight_matrix[:, 2, :], weight_matrix[:, 6, :] = 0.05, 0.05
    # weight_matrix[:, 3, :], weight_matrix[:, 7, :] = 0.05, 0.05
    # output = output * weight_matrix
    # self_softmax = F.softmax(torch.stack([refine_ema_logits, refine_check_logits], dim=1), dim=-1)
    # output = torch.cat([output, 0.25 * self_softmax], dim=1)

    # # 加权-k=3
    weight_matrix = torch.ones_like(output, device='cuda')
    weight_matrix[:, 0, :], weight_matrix[:, 3, :] = 0.1, 0.1
    weight_matrix[:, 1, :], weight_matrix[:, 4, :] = 0.05, 0.05
    weight_matrix[:, 2, :], weight_matrix[:, 5, :] = 0.05, 0.05
    output = output * weight_matrix
    self_softmax = F.softmax(torch.stack([refine_ema_logits, refine_check_logits], dim=1), dim=-1)
    output = torch.cat([output, 0.3 * self_softmax], dim=1)

    # 加权-k=2
    # weight_matrix = torch.ones_like(output, device='cuda')
    # weight_matrix[:, 0, :], weight_matrix[:, 2, :] = 0.1, 0.1
    # weight_matrix[:, 1, :], weight_matrix[:, 3, :] = 0.1, 0.1
    # output = output * weight_matrix
    # self_softmax = F.softmax(torch.stack([refine_ema_logits, refine_check_logits], dim=1), dim=-1)
    # output = torch.cat([output, 0.3 * self_softmax], dim=1)

    # 加权-k=1
    # weight_matrix = torch.ones_like(output, device='cuda')
    # weight_matrix[:, 0, :], weight_matrix[:, 1, :] = 0.2, 0.2
    # output = output * weight_matrix
    # self_softmax = F.softmax(torch.stack([refine_ema_logits, refine_check_logits], dim=1), dim=-1)
    # output = torch.cat([output, 0.3 * self_softmax], dim=1)
    # refine pseudo labels
    avg_softmax = torch.sum(output, dim=1)
    # abs to log_indices
    avg_softmax[log_indices] = self_softmax[:, 0, :][log_indices]
    # 如果出现负 refine，则取消该pixel的refine
    raw_softmax = self_softmax[:, 0, :]
    avg_, _ = torch.topk(avg_softmax, k=2, dim=-1)
    avg_diff = (avg_[:, 0] - avg_[:, 1]) / avg_[:, 0]
    raw_, _ = torch.topk(raw_softmax, k=2, dim=-1)
    raw_diff = (raw_[:, 0] - raw_[:, 1]) / raw_[:, 0]
    _, raw_labels = torch.max(raw_softmax, dim=-1)
    _, new_labels = torch.max(avg_softmax, dim=-1)
    condition = raw_diff > avg_diff
    new_labels[condition] = raw_labels[condition]
    avg_softmax[condition] = raw_softmax[condition]
    return new_labels

@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.source_only = cfg['source_only']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None
        self.memory_bank = None
        self.warm_iter = 100
        self.end_iter = 4000
        self.k = 3
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,  # 设置日志记录的级别为INFO
            format='%(asctime)s - %(message)s',  # 设置日志格式
            filename='MICCCGTA.log',  # 设置日志文件名
            filemode='w'  # 设置文件模式为写入模式
        )
        self.logger = logging.getLogger('MICCCGTA.log')
        self.sum = 0



    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_label,
                      target_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        prepare_dict = {}
        # TODO stage 1 获得teacher在源域和目标域的预测
        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        # ema_logits正常做伪标签生成，ema_target_emb为老师在目标域的emb
        ema_logits, ema_target_logits = self.get_ema_model().generate_pseudo_label_with_scale(
            target_img, target_img_metas, up=True)
        # 原始操作
        seg_debug['Target'] = self.get_ema_model().decode_head.debug_output
        # 中间较不可信的像素点会重新赋值
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight1 = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight1 * torch.ones(
            pseudo_prob.shape, device=dev)

        dis = self.local_iter / self.max_iters
        before_refine = (target_label.squeeze(1) == pseudo_label).sum()
        if self.local_iter % 100 == 0:
            self.logger.info(before_refine)
        # TODO stage2 获得学生在目标域上的预测和emb，这个既可以用于方法一也能用于方法二
        # check_logits, model_target_logits, model_target_emb = self.get_model().generate_pseudo_label_with_proj(
        #     target_img, target_img_metas, up=True)
        # # TODO： 方法一
        # if self.local_iter >= self.warm_iter and self.local_iter <= self.end_iter:
        #     # 开始refine的时候，最可信部分的pw值应当提高,refine的像素中假设只有10%可以稳定正确
        #     value = (ps_size - torch.sum(ps_large_p).item()) * dis / ps_size
        #     up_value = 0.1 * value
        #     pseudo_weight[ps_large_p] += up_value
        #     B, H, W = pseudo_label.shape
        #
        #     one_minus_mask = ~ps_large_p  # 通过硬阈值的mask取反
        #     discard_logits = pseudo_prob[one_minus_mask]  # 没通过的pixels
        #     alpha_value = torch.quantile(
        #         discard_logits, 1 - pow(dis, 0.2))  # 取对应的阈值
        #     ps_large_alpha = pseudo_prob.le(alpha_value).long() == 1  # 找出大于这个阈值的所有像素点，会包含ps_large_p
        #     uncertain_pixels_idx = (ps_large_alpha & one_minus_mask).reshape(-1).reshape(-1)  # 大于动态阈值，小于硬阈值的mask
        #     del ps_large_alpha
        #     # 再分别让老师和学生取出这些pixels和memorybank进行余弦相似度距离计算，得到最相似的alpha个特征
        #     refine_ema_logits = ema_logits.permute(0, 2, 3, 1).reshape(B * H * W, -1)[uncertain_pixels_idx]
        #     refine_check_logits = check_logits.permute(0, 2, 3, 1).reshape(B * H * W, -1)[uncertain_pixels_idx]
        #     new_labels = compute_neighbors(refine_ema_logits, refine_check_logits,
        #                                    self.memory_bank,
        #                                    self.k,
        #                                    dis)
        #     # 计算余弦相似度, 找到最近的三个，refine伪标签,并给与权重
        #     pseudo_label_refine = pseudo_label.clone()
        #     pseudo_label_refine = pseudo_label_refine.reshape(-1)
        #     pseudo_label_refine[[uncertain_pixels_idx]] = new_labels
        #     pseudo_label_refine = pseudo_label_refine.reshape(B, H, W)
        #     pseudo_label = pseudo_label_refine
        #     after_refine = (target_label.squeeze(1) == pseudo_label_refine).sum()
        #     total_refine = after_refine - before_refine
        #     if total_refine > 0:
        #         pseudo_label = pseudo_label_refine
        #     self.sum += total_refine
        #     if self.local_iter % 100 == 0:
        #         self.logger.info(f"{self.local_iter} iter total refine {self.sum}")
        #         self.sum = 0
        #     del check_logits, uncertain_pixels_idx, refine_ema_logits, refine_check_logits, pseudo_label_refine
        # del pseudo_prob, ps_large_p, ps_size, ema_logits, pseudo_weight1

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # emb_target_entropy = entropy(ema_softmax, dim=1)
        # low_thresh = np.percentile(emb_target_entropy.cpu().numpy().flatten(), 50)
        # low_entropy_mask = (emb_target_entropy.le(low_thresh).float().bool())
        # low_mask_all = torch.cat(  # 把标签数据只要不为255的和无标签数据刚刚算出来的高低掩码矩阵分别cat起来
        #     ((gt_semantic_seg != 255).float(),
        #      low_entropy_mask.unsqueeze(1)))
        # low_mask_all = F.interpolate(
        #     low_mask_all, size=model_target_emb.shape[2:], mode="nearest"
        # )
        # # TODO stage3 在源域中计算对比损失
        # prepare_dict['st_target_emb'] = model_target_emb
        # prepare_dict['st_target_logits'] = model_target_logits
        # prepare_dict['pseudo_label_small'] = F.interpolate(label_onehot(pseudo_label, self.num_classes),
        #                                                    size=model_target_emb.shape[2:],
        #                                                    mode="nearest")
        # prepare_dict['te_target_prob'] = F.interpolate(F.softmax(ema_target_logits, dim=1),
        #                                                 size=model_target_emb.shape[-2:],
        #                                                 mode='bilinear',
        #                                                 align_corners=False)
        # prepare_dict['low_mask_all'] = low_mask_all
        # clean_losses = self.get_model().forward_train_source(
        #     img, img_metas, gt_semantic_seg, return_feat=True, prepare_dict=prepare_dict
        # )
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True
        )

        # self.memory_bank = clean_losses.pop('decode.memorybank')
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().decode_head.debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        # clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')
        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            # feat_loss.backward()
            clean_loss = clean_loss + feat_loss
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        clean_loss.backward()
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mixed_seg_weight = pseudo_weight.clone()
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack(
                    (gt_semantic_seg[i][0], pseudo_label[i])))
            _, mixed_seg_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        del gt_pixel_weight
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=False)
        seg_debug['Mix'] = self.get_model().decode_head.debug_output
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        # Masked Training
        if self.enable_masking and self.mask_mode.startswith('separate'):
            masked_loss = self.mic(self.get_model(), img, img_metas,
                                   gt_semantic_seg, target_img,
                                   target_img_metas, valid_pseudo_mask,
                                   pseudo_label, pseudo_weight)
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()

        # if self.local_iter % self.debug_img_interval == 0 and \
        #         not self.source_only:
        #     out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
        #     os.makedirs(out_dir, exist_ok=True)
        #     vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
        #     vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
        #     vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
        #     for j in range(batch_size):
        #         rows, cols = 2, 5
        #         fig, axs = plt.subplots(
        #             rows,
        #             cols,
        #             figsize=(3 * cols, 3 * rows),
        #             gridspec_kw={
        #                 'hspace': 0.1,
        #                 'wspace': 0,
        #                 'top': 0.95,
        #                 'bottom': 0,
        #                 'right': 1,
        #                 'left': 0
        #             },
        #         )
        #         subplotimg(axs[0][0], vis_img[j], 'Source Image')
        #         subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
        #         subplotimg(
        #             axs[0][1],
        #             gt_semantic_seg[j],
        #             'Source Seg GT',
        #             cmap='cityscapes')
        #         subplotimg(
        #             axs[1][1],
        #             pseudo_label[j],
        #             'Target Seg (Pseudo) GT',
        #             cmap='cityscapes')
        #         subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
        #         subplotimg(
        #             axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
        #         # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
        #         #            cmap="cityscapes")
        #         if mixed_lbl is not None:
        #             subplotimg(
        #                 axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
        #         subplotimg(
        #             axs[0][3],
        #             mixed_seg_weight[j],
        #             'Pseudo W.',
        #             vmin=0,
        #             vmax=1)
        #         if self.debug_fdist_mask is not None:
        #             subplotimg(
        #                 axs[0][4],
        #                 self.debug_fdist_mask[j][0],
        #                 'FDist Mask',
        #                 cmap='gray')
        #         if self.debug_gt_rescale is not None:
        #             subplotimg(
        #                 axs[1][4],
        #                 self.debug_gt_rescale[j],
        #                 'Scaled GT',
        #                 cmap='cityscapes')
        #         for ax in axs.flat:
        #             ax.axis('off')
        #         plt.savefig(
        #             os.path.join(out_dir,
        #                          f'{(self.local_iter + 1):06d}_{j}.png'))
        #         plt.close()
        #
        # if self.local_iter % self.debug_img_interval == 0:
        #     out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
        #     os.makedirs(out_dir, exist_ok=True)
        #     if seg_debug['Source'] is not None and seg_debug:
        #         if 'Target' in seg_debug:
        #             seg_debug['Target']['Pseudo W.'] = mixed_seg_weight.cpu(
        #             ).numpy()
        #         for j in range(batch_size):
        #             cols = len(seg_debug)
        #             rows = max(len(seg_debug[k]) for k in seg_debug.keys())
        #             fig, axs = plt.subplots(
        #                 rows,
        #                 cols,
        #                 figsize=(5 * cols, 5 * rows),
        #                 gridspec_kw={
        #                     'hspace': 0.1,
        #                     'wspace': 0,
        #                     'top': 0.95,
        #                     'bottom': 0,
        #                     'right': 1,
        #                     'left': 0
        #                 },
        #                 squeeze=False,
        #             )
        #             for k1, (n1, outs) in enumerate(seg_debug.items()):
        #                 for k2, (n2, out) in enumerate(outs.items()):
        #                     subplotimg(
        #                         axs[k2][k1],
        #                         **prepare_debug_out(f'{n1} {n2}', out[j],
        #                                             means, stds))
        #             for ax in axs.flat:
        #                 ax.axis('off')
        #             plt.savefig(
        #                 os.path.join(out_dir,
        #                              f'{(self.local_iter + 1):06d}_{j}_s.png'))
        #             plt.close()
        #         del seg_debug
        self.local_iter += 1

        return log_vars
