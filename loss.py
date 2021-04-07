import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
import random


class DCLoss(nn.Module):
    

    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin = margin)
        self.intra_loss = nn.MarginRankingLoss(margin = 0.1)
    def forward(self, inputs, targets):
        n = inputs.size(0)

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss_tri = self.ranking_loss(dist_an, dist_ap, y)


        feat_V, feat_T = torch.chunk(inputs, 2, dim=0)
        feat_all_V = torch.cat((feat_V, feat_V), dim = 0)
        dist_V = torch.pow(feat_all_V, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_V = dist_V + dist_V.t()
        dist_V.addmm_(1, -2, feat_all_V, feat_all_V.t())
        dist_V = dist_V.clamp(min=1e-12).sqrt() 
        mask_V = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap_V, dist_an_V = [], []
        for i in range(n):
            dist_ap_V.append(dist_V[i][mask_V[i]].max().unsqueeze(0))
            dist_an_V.append(dist_V[i][mask_V[i] == 0].min().unsqueeze(0))
        dist_ap_V = torch.cat(dist_ap_V)
        dist_an_V = torch.cat(dist_an_V)

        y = torch.ones_like(dist_an_V)
        loss_V_tri = self.intra_loss(dist_an_V, dist_ap_V, y)

        feat_all_T = torch.cat((feat_T, feat_T), dim=0)
        dist_T = torch.pow(feat_all_T, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_T = dist_T + dist_T.t()
        dist_T.addmm_(1, -2, feat_all_T, feat_all_T.t())
        dist_T = dist_T.clamp(min=1e-12).sqrt() 
        mask_T = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap_T, dist_an_T = [], []
        for i in range(n):
            dist_ap_T.append(dist_T[i][mask_T[i]].max().unsqueeze(0))
            dist_an_T.append(dist_T[i][mask_T[i] == 0].min().unsqueeze(0))
        dist_ap_T = torch.cat(dist_ap_T)
        dist_an_T = torch.cat(dist_an_T)
        y = torch.ones_like(dist_an_T)
        loss_T_tri = self.intra_loss(dist_an_T, dist_ap_T, y)

        loss = loss_V_tri + loss_T_tri 
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct
