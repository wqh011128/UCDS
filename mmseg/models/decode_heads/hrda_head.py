# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add return_logits flag
# - Update debug_output
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy
from abc import ABC
from collections import deque

import torch
from torch.nn import functional as F
import torch.nn as nn

from ...core import add_prefix
from ...ops import resize as _resize
from .. import builder
from ..builder import HEADS
from ..segmentors.hrda_encoder_decoder import crop
from .decode_head import BaseDecodeHead
from mmseg.utils.memorybank import ReliableBank
def label_onehot(inputs, num_segments):
    if inputs.ndim == 4:
        inputs = inputs.squeeze(1)
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_segments, batch_size, im_h, im_w)).cuda()

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    outputs.scatter_(0, inputs_temp.unsqueeze(1), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)
def scale_box(box, scale):
    y1, y2, x1, x2 = box
    # assert y1 % scale == 0
    # assert y2 % scale == 0
    # assert x1 % scale == 0
    # assert x2 % scale == 0
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


def proto_criterion(source_emb, target_emb, gt_souce, pseudo_label):
    C = pseudo_label.shape[1]
    D = source_emb.shape[1]
    source_emb = source_emb.permute(0, 2, 3, 1)
    target_emb = target_emb.permute(0, 2, 3, 1)
    source_emb_proto = torch.zeros((C, D)).cuda()
    target_emb_proto = torch.zeros((C, D)).cuda()
    for i in range(C):
        index_i_source = gt_souce[:, i].bool()
        index_i_target = pseudo_label[:, i].bool()
        if torch.any(index_i_source).item() is True and torch.any(index_i_target).item() is True:
            source_emb_proto[i] = source_emb[index_i_source].mean(dim=0)
            target_emb_proto[i] = target_emb[index_i_target].mean(dim=0)
    loss = F.mse_loss(source_emb_proto, target_emb_proto, reduction='sum') / D
    return loss



def contrast_criterion(emb, model_source_logits, gt_semantic_seg, prepare_dict, i_iter, momentum=None):
    st_emb = torch.cat([emb, prepare_dict['st_target_emb']], dim=0)
    te_target_prob = prepare_dict['te_target_prob']
    st_source_prob = F.interpolate(F.softmax(model_source_logits, dim=1),
                                      size=emb.shape[-2:],
                                      mode='bilinear',
                                      align_corners=False)
    low_mask_all = prepare_dict['low_mask_all']

    pseudo_label_small = prepare_dict['pseudo_label_small']
    source_label_small = F.interpolate(label_onehot(gt_semantic_seg, pseudo_label_small.shape[1]),
                                       size=emb.shape[-2:],
                                       mode='nearest').long()
    current_class_threshold = 0.2
    low_rank, high_rank = 3, 20
    temp = 0.1
    num_queries = 256
    num_negatives = 100
    num_feat = st_emb.shape[1]  # proj的维度

    num_segments = source_label_small.shape[1]  # 类别个数
    # 将学生生成的源域和目标域标签乘以源域掩码和目标域掩码，低熵掩码和高熵掩码
    low_entropy_label = torch.cat((source_label_small, pseudo_label_small), dim=0) * low_mask_all  # 筛选出学生生成的预测中，有标签的和无标签的低熵像素点，这里乘以01，所以保持矩阵形式
    # 学生和教师的proj
    rep = st_emb.permute(0, 2, 3, 1)

    seg_feat_low_entropy_list = []  # candidate anchor pixels
    seg_num_list = []  # the number of low_valid pixels in each class
    seg_proto_list = []  # the center of each class

    _, prob_sort_source = torch.sort(st_source_prob, 1, True)
    prob_sort_source = prob_sort_source.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

    _, prob_sort_target = torch.sort(te_target_prob, 1, True)
    prob_sort_target = prob_sort_target.permute(0, 2, 3, 1)  # (num_unlabeled, h, w, num_cls)
    prob = torch.cat((st_source_prob, te_target_prob), dim=0)  # (batch_size, num_cls, h, w)#再把老师预测的cat起来

    valid_classes = []
    neg_list = []
    for i in range(num_segments):  # 对于每一个类别
        low_entropy_maski = low_entropy_label[:, i]

        prob_seg = prob[:, i, :, :]  # 老师预测的有标签和吴标签的第i个通道softmax值，也就是i类
        rep_mask_low_entropy_i = (  # 老师在i类的softmax值大于阈值且对学生来说也是在i类上低熵
                                       prob_seg > current_class_threshold
                               ) * low_entropy_maski.bool()
        # 上面只存学生在i类的学生认为低熵的pixels，下面除了上面的条件外还限制了老师对i类的softmax值大于阈值，但用的都是学生proj
        seg_feat_low_entropy_list.append(rep[low_entropy_maski.bool()])  # 第i类低熵的同时softmax值要高于一定阈值，才会被选中
        # positive sample: center of the class
        seg_proto_list.append(
            torch.mean(
                rep[rep_mask_low_entropy_i], dim=0, keepdim=True
            )
        )

        class_mask_u = torch.sum(
            prob_sort_target[:, :, :, low_rank:high_rank].eq(i), dim=3
        ).bool()  # 老师生成的伪标签中给与top-lowrank较高的相信度，剩下所有类别都不相信，然后在所有点中选出存在i的像素点，经过sum变成掩码
        # TODO：即对于i类来说，class_mask_u就是这张图中每一个像素点来说，不能相信i类别的像素点（对该pixel来说i类是置信度不在topk之内的）
        class_mask_l = torch.sum(prob_sort_source[:, :, :, :low_rank].eq(i), dim=3).bool()  # 同理选出了标签数据中i在topk之内的像素点

        negative_mask = torch.cat(  # 将i在topk但不是最可信的pixels选出来，再和不可信cat起来，组成了标签和无标签数据中
            (class_mask_l * (source_label_small[:, i] == 0), class_mask_u), dim=0  # 对i类来说，都不是最相信的掩码
        )

        # 该掩码中，标签data表示熵高的pixels中第i类softmax小于阈值，且还不是算出来最可信的点，无标签data表示熵高的pixels中第i类softmax小于阈值且对该pixel来说i类是置信度不在topk之内的
        keys = rep[negative_mask]  # 老师的proj作为负样本
        neg_list.append(keys)

        if rep_mask_low_entropy_i.sum() > 0:  # 存在i类低熵的像素点
            seg_num_list.append(int(rep_mask_low_entropy_i.sum().item()))  # 学生只要认为有低熵的就行
            valid_classes.append(i)  # 记录class
    if (len(seg_num_list) <= 1):
        return torch.tensor(0.0).cuda() * rep.sum(), momentum
    else:
        reco_loss = torch.tensor(0.0).cuda()
        seg_proto = torch.cat([seg_proto_list[i] for i in valid_classes], dim=0)  # 每个类别低熵的教师原型  # shape: [valid_seg, 256]
        valid_seg = len(seg_num_list)  # 一共有多少类别可以计算对比损失？  # number of valid classes
        # 记录新的原型
        prototype = torch.zeros(
            (prob_sort_source.shape[-1], num_queries, 1, num_feat)
        ).cuda()
        count = 0
        for i in range(valid_seg):  # 对每个类别
            if (
                    len(seg_feat_low_entropy_list[valid_classes[i]]) > 0  # 有锚点和负样本？
                    and len(neg_list[valid_classes[i]]) > 0
            ):
                # select anchor pixel anchor用的是学生proj中学生认为低熵且老师认为i类softmax大于阈值的
                seg_low_entropy_idx = torch.randint(  # 果然是选择锚点，低熵用来选择锚点
                    len(seg_feat_low_entropy_list[valid_classes[i]]), size=(num_queries,)
                )
                anchor_feat = (
                    seg_feat_low_entropy_list[valid_classes[i]][seg_low_entropy_idx]
                )
            else:
                # in some rare cases, all queries in the current query class are easy
                reco_loss = reco_loss + 0 * rep.sum()
                count += 1
                continue

            negative_feat = neg_list[valid_classes[i]]

            high_entropy_idx = torch.randint(
                len(negative_feat), size=(num_queries * num_negatives,)
            )
            negative_feat = negative_feat[high_entropy_idx]
            negative_feat = negative_feat.reshape(  # 每一个类有num_query个查询
                num_queries, num_negatives, num_feat
            )
            positive_feat = (
                seg_proto[i]  # 正样本只有一个教师原型
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(num_queries, 1, 1)
                .cuda()
            )  # (num_queries, 1, num_feat)

            if momentum is not None:
                ema_decay = min(1 - 1 / (i_iter + 1), 0.5)
                positive_feat = (1 - ema_decay) * positive_feat + ema_decay * momentum[valid_classes[i]]
                prototype[valid_classes[i]] = positive_feat.detach()
            all_feat = torch.cat(
                (positive_feat, negative_feat), dim=1
            )  # (num_queries, 1 + num_negative, num_feat)

            seg_logits = torch.cosine_similarity(
                anchor_feat.unsqueeze(1), all_feat, dim=2
            )

            reco_loss = reco_loss + F.cross_entropy(
                seg_logits / temp, torch.zeros(num_queries).long().cuda()
            )
        if count == valid_seg:
            return torch.tensor(0.0).cuda(), momentum
        elif momentum is not None:
            return reco_loss / valid_seg, prototype
        return reco_loss / valid_seg, momentum


@HEADS.register_module()
class HRDAHead(BaseDecodeHead):

    def __init__(self,
                 single_scale_head,
                 lr_loss_weight=0,
                 hr_loss_weight=0,
                 scales=[1],
                 attention_embed_dim=256,
                 attention_classwise=True,
                 enable_hr_crop=False,
                 hr_slide_inference=True,
                 fixed_attention=None,
                 debug_output_attention=False,
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)
        if single_scale_head == 'DAFormerHead':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        elif single_scale_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        else:
            raise NotImplementedError(single_scale_head)
        super(HRDAHead, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        head_cfg['type'] = single_scale_head
        self.head = builder.build_head(head_cfg)

        attn_cfg['type'] = single_scale_head
        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.scale_attention = builder.build_head(attn_cfg)
        else:
            self.scale_attention = None
            self.fixed_attention = fixed_attention
        self.lr_loss_weight = lr_loss_weight
        self.hr_loss_weight = hr_loss_weight
        self.scales = scales
        self.enable_hr_crop = enable_hr_crop
        self.hr_crop_box = None
        self.hr_slide_inference = hr_slide_inference
        self.debug_output_attention = debug_output_attention
        # Contrast Loss Add
        self.loss_weight1 = 0.3
        # self.loss_weight2 = 0.1
        self.i_iter = 0
        self.memory_bank = ReliableBank(dim=19, class_num=19, memory_length=1000)
        self.momentum = torch.zeros((19, 256, 1, 256), requires_grad=False).cuda()

    def set_hr_crop_box(self, boxes):
        self.hr_crop_box = boxes

    def hr_crop_slice(self, scale):
        crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.align_corners)

    def decode_hr(self, inp, bs):
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            features = inp['features']  # level, crop * bs, c, h, w
            boxes = inp['boxes']
            dev = features[0][0].device
            h_img, w_img = 0, 0
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                device=dev)
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)

            crop_seg_logits = self.head(features)
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            return preds
        else:
            return self.head(inp)

    def get_scale_attention(self, inp):
        if self.scale_attention is not None:
            att = torch.sigmoid(self.scale_attention(inp))
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs):
        assert len(inputs) == 2
        hr_inp = inputs[1]
        hr_scale = self.scales[1]
        lr_inp = inputs[0]
        lr_sc_att_inp = inputs[0]  # separate var necessary for stack hr_fusion
        lr_scale = self.scales[0]
        batch_size = lr_inp[0].shape[0]
        assert lr_scale <= hr_scale

        has_crop = self.hr_crop_box is not None
        if has_crop:
            crop_y1, crop_y2, crop_x1, crop_x2 = self.hr_crop_box

        # print_log(f'lr_inp {[f.shape for f in lr_inp]}', 'mmseg')
        lr_seg = self.head(lr_inp)
        # print_log(f'lr_seg {lr_seg.shape}', 'mmseg')

        hr_seg = self.decode_hr(hr_inp, batch_size)

        att = self.get_scale_attention(lr_sc_att_inp)
        if has_crop:
            mask = lr_seg.new_zeros([lr_seg.shape[0], 1, *lr_seg.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os)
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask
        # print_log(f'att {att.shape}', 'mmseg')
        lr_seg = (1 - att) * lr_seg
        # print_log(f'scaled lr_seg {lr_seg.shape}', 'mmseg')
        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)
        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)

        if has_crop:
            hr_seg_inserted = torch.zeros_like(up_lr_seg)
            slc = self.hr_crop_slice(self.os)
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg
        else:
            hr_seg_inserted = hr_seg

        fused_seg = att * hr_seg_inserted + up_lr_seg

        if self.debug_output_attention:
            att = torch.sum(
                att * torch.softmax(fused_seg, dim=1), dim=1, keepdim=True)
            return att, None, None

        if self.debug:
            self.debug_output.update({
                'High Res':
                torch.max(hr_seg, dim=1)[1].detach().cpu().numpy(),
                'High Res Inserted':
                torch.max(hr_seg_inserted, dim=1)[1].detach().cpu().numpy(),
                'Low Res':
                torch.max(lr_seg, dim=1)[1].detach().cpu().numpy(),
                'Fused':
                torch.max(fused_seg, dim=1)[1].detach().cpu().numpy(),
            })
            if torch.is_tensor(att):
                self.debug_output['Attention'] = torch.sum(
                    att * torch.softmax(fused_seg, dim=1), dim=1,
                    keepdim=True).detach().cpu().numpy()

        return fused_seg, lr_seg, hr_seg

    def reset_crop(self):
        del self.hr_crop_box
        self.hr_crop_box = None

    def forward_train_source(self,
                             inputs,
                             img_metas,
                             emb,
                             gt_semantic_seg,
                             train_cfg,
                             seg_weight=None,
                             prepare_dict=None):
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        losses['con_loss'], self.momentum = contrast_criterion(emb,
                                                               seg_logits[0].detach(),
                                                               gt_semantic_seg,
                                                               prepare_dict,
                                                               self.i_iter,
                                                               self.momentum)
        losses['con_loss'] = self.loss_weight1 * losses['con_loss']
        # 用fuse_seg更新bank，在生成伪标签的时候做refine
        memorybank = self.memory_bank.update_(seg_logits[0].detach(), gt_semantic_seg)
        losses['memorybank'] = memorybank
        # losses['proto_loss'] = self.loss_weight2 * proto_criterion(
        #     emb,
        #     prepare_dict['st_target_emb'],
        #     F.interpolate(
        #         label_onehot(gt_semantic_seg,
        #                      prepare_dict['pseudo_label_small'].shape[1]),
        #                      size=emb.shape[-2:],
        #                      mode='nearest'
        #     ).long(),
        #     prepare_dict['pseudo_label_small']
        # )
        self.i_iter += 1
        self.reset_crop()
        return losses

    def forward_train_mix(self,
                          inputs,
                          img_metas,
                          gt_semantic_seg,
                          train_cfg,
                          seg_weight=None):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        seg_logits = self.forward(inputs)
        # 用fuse_seg更新bank，在生成伪标签的时候做refine
        # memorybank = self.memory_bank.update_(seg_logits[0].detach(), gt_semantic_seg, seg_weight)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        # losses['memorybank'] = memorybank

        self.reset_crop()
        return losses

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      return_logits=False):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        if return_logits:
            losses['logits'] = seg_logits
        self.reset_crop()
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute losses."""
        fused_seg, lr_seg, hr_seg = seg_logit
        loss = super(HRDAHead, self).losses(fused_seg, seg_label, seg_weight)
        if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
            return loss

        if self.lr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(lr_seg, seg_label,
                                                 seg_weight), 'lr'))
        if self.hr_loss_weight > 0 and self.enable_hr_crop:
            cropped_seg_label = crop(seg_label, self.hr_crop_box)
            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
            else:
                cropped_seg_weight = seg_weight
            if self.debug:
                self.debug_output['Cropped GT'] = \
                    cropped_seg_label.squeeze(1).detach().cpu().numpy()
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, cropped_seg_label,
                                                 cropped_seg_weight), 'hr'))
        elif self.hr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, seg_label,
                                                 seg_weight), 'hr'))
        loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
        if self.lr_loss_weight > 0:
            loss['lr.loss_seg'] *= self.lr_loss_weight
        if self.hr_loss_weight > 0:
            loss['hr.loss_seg'] *= self.hr_loss_weight

        if self.debug:
            self.debug_output['GT'] = \
                seg_label.squeeze(1).detach().cpu().numpy()
            # Remove debug output from cross entropy loss
            self.debug_output.pop('Seg. Pred.', None)
            self.debug_output.pop('Seg. GT', None)

        return loss
