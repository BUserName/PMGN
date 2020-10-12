import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function


class ProjectNetwork(nn.Module):
    def __init__(self, args, src_fea, tar_fea):
        super(ProjectNetwork, self).__init__()
        self.src_feature = src_fea
        self.tar_feature = tar_fea
        self.dropout = args.dropout

        self.transform_src = nn.Sequential(
            nn.Linear(self.src_feature, 1024),
            nn.Linear(1024, 512),
            nn.Dropout(p=self.dropout),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )#
        self.transform_tar = nn.Sequential(
            nn.Linear(self.tar_feature, 1024),
            nn.Linear(1024, 512),
            nn.Dropout(p=self.dropout),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )#

    def forward(self, src_feat, tar_feat):
        num_task = src_feat.size(0)
        num_data = src_feat.size(1)
        src_feat = src_feat.view(num_data * num_task, src_feat.size(2))
        tar_feat = tar_feat.view(num_data * num_task, tar_feat.size(2))

        src_feat_1 = self.transform_src(src_feat)
        tar_feat_1 = self.transform_tar(tar_feat)

        src_feat = src_feat_1.view(num_task, num_data, src_feat_1.size(1))
        tar_feat = tar_feat_1.view(num_task, num_data, tar_feat_1.size(1))

        combine_fea = torch.cat([src_feat, tar_feat], axis=1).view(num_data*num_task*2, src_feat.size(2))
        return src_feat_1, tar_feat_1, combine_fea


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, args.num_class)
        )

    def forward(self, src_feat, tar_feat):
        src_pred = self.classifier(src_feat)
        tar_pred = self.classifier(tar_feat)

        return src_pred, tar_pred