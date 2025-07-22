# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
from collections import deque
import torch.nn.functional as F
import random
def entropy(p, dim=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim)

class ReliableBank:
    def __init__(self, dim, class_num, memory_length=100):
        super(ReliableBank, self).__init__()
        self.dim = dim
        self.class_num = class_num
        self.ignore_class = 255
        self.bank = torch.zeros(memory_length, self.dim).cuda()
        self.MemoryBank = [self.bank for _ in range(self.class_num)]
        self.RefineBank = [self.bank for _ in range(self.class_num)]
        self.memory_lens = memory_length
        self.max_keep = 500

    @torch.no_grad()
    def update_(self, logits, labels):
        """
        features:B,C,H,W
        labels:B,1,H,W
        note: 255, entropy
        """
        bank = dict()
        B, C, H, W = logits.shape
        labels = F.interpolate(labels.float(), size=logits.shape[2:], mode='nearest').long().squeeze(1)
        filter_ignore_idx = torch.where(labels != self.ignore_class, 1, 0).bool()
        # 过滤不可能的像素
        labels = labels[filter_ignore_idx]
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)[filter_ignore_idx.reshape(-1)]
        # 求出mix domain的prob以及预测标签
        probs = F.softmax(logits, dim=-1)
        pred_prob, pred_labels = torch.max(probs, dim=-1)
        unique_labels = torch.unique(pred_labels)
        logits_clone = logits.clone()
        for c in unique_labels:
            # 对于每一个类别c,计算简单样本的索引
            easy_index_c = ((labels == c) & (pred_labels == c)).nonzero(as_tuple=False).squeeze(-1)
            if len(easy_index_c) > 1:
                easy_entropy = entropy(probs[easy_index_c])
                alpha = torch.quantile(easy_entropy, 0.7)
                mask = easy_entropy.ge(alpha).long() == 1
                easy_index_c = easy_index_c[mask].unsqueeze(-1)
            else:
                easy_index_c = easy_index_c.unsqueeze(-1)

            hard_index_c = ((labels == c) & (pred_labels != c)).nonzero(as_tuple=False).squeeze(-1)
            if len(hard_index_c) > 1:
                hard_entropy = entropy(probs[hard_index_c])
                alpha = torch.quantile(hard_entropy, 0.7)
                mask = hard_entropy.ge(alpha).long() == 1
                hard_index_c = hard_index_c[mask].unsqueeze(-1)
            else:
                hard_index_c = hard_index_c.unsqueeze(-1)
            # 如果class c没有已经准确预测的pixels，就跳过，不会加入到bank中
            if easy_index_c.ndim == 0 and hard_index_c.ndim == 0:
                continue
            num_easy = easy_index_c.shape[0]
            num_hard = hard_index_c.shape[0]
            if num_hard >= 9 * self.max_keep / 10 and num_easy >= self.max_keep / 10:
                hard = 9 * self.max_keep // 10
                easy = self.max_keep - hard
            elif num_easy < self.max_keep / 10 and num_hard >= (self.max_keep - num_easy):
                easy = num_easy
                hard = self.max_keep - easy
            elif num_hard < 9 * self.max_keep / 10 and num_easy >= (self.max_keep - num_hard):
                hard = num_hard
                easy = self.max_keep - hard
            else:
                hard = num_hard
                easy = num_easy
            hard_index_c = hard_index_c[torch.randperm(num_hard)[:hard]]
            easy_index_c = easy_index_c[torch.randperm(num_easy)[:easy]]
            index_c = torch.cat((hard_index_c, easy_index_c), dim=0).squeeze(-1)
            self.MemoryBank[c] = torch.cat([self.MemoryBank[c], logits[index_c]], dim=0)[-self.memory_lens:]
            # 交换困难样本中softmax的最大值和c类，即swap
            if hard_index_c.shape[0] != 0:
                prob_hard = torch.softmax(logits_clone[hard_index_c.squeeze(-1)], dim=-1)
                _, hard_max_idx = torch.max(prob_hard, dim=-1)
                logits_clone[hard_index_c.squeeze(-1), c.item()] = logits[hard_index_c.squeeze(-1), hard_max_idx]
                logits_clone[hard_index_c.squeeze(-1), hard_max_idx] = logits[hard_index_c.squeeze(-1), c.item()]
                self.RefineBank[c] = torch.cat([self.RefineBank[c], logits_clone[index_c]], dim=0)[-self.memory_lens:]
            else:
                self.RefineBank[c] = torch.cat([self.RefineBank[c], logits[index_c]], dim=0)[-self.memory_lens:]
        bank['memory_bank'] = self.MemoryBank
        bank['refine_bank'] = self.RefineBank
        return bank

