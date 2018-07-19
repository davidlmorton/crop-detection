from sconce.models.base import Model
from sconce.models.layers import Convolution2dLayer, AdaptiveAveragePooling2dLayer, FullyConnectedLayer
from torch import nn

import numpy as np
import torch


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class YModel(Model):
    def __init__(self, base_model):
        super().__init__()

        self.base_model = base_model

        self.feature_extractor = nn.Sequential(
            self.base_model.conv1,
            self.base_model.bn1,
            self.base_model.relu,
            self.base_model.maxpool,

            self.base_model.layer1,
            self.base_model.layer2,
            self.base_model.layer3,
            self.base_model.layer4,
        )

        pool_size = 4
        self.is_paired_classifier = nn.Sequential(
            Convolution2dLayer(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1),
            Convolution2dLayer(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1),
            AdaptiveAveragePooling2dLayer(in_channels=64, output_size=pool_size, activation=None),
            Flatten(),
            FullyConnectedLayer(in_size=64*pool_size*pool_size, out_size=32, activation=nn.ReLU(inplace=True)),
            FullyConnectedLayer(in_size=32, out_size=2, activation=None),
            nn.LogSoftmax(dim=1),
        )

        self.is_swapped_classifier = nn.Sequential(
            Convolution2dLayer(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1),
            Convolution2dLayer(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1),
            AdaptiveAveragePooling2dLayer(in_channels=64, output_size=pool_size, activation=None),
            Flatten(),
            FullyConnectedLayer(in_size=64*pool_size*pool_size, out_size=32, activation=nn.ReLU(inplace=True)),
            FullyConnectedLayer(in_size=32, out_size=2, activation=None),
            nn.LogSoftmax(dim=1),
        )

        self.xy_regressor = nn.Sequential(
            Convolution2dLayer(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1),
            Convolution2dLayer(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1),
            AdaptiveAveragePooling2dLayer(in_channels=64, output_size=pool_size, activation=None),
            Flatten(),
            FullyConnectedLayer(in_size=64*pool_size*pool_size, out_size=32, activation=nn.ReLU(inplace=True)),
            FullyConnectedLayer(in_size=32, out_size=2, activation=None),
            nn.Sigmoid(),
        )
        self.alpha = 0.0

    def forward(self, inputs, **kwargs):
        left_images = inputs[:, :3]
        right_images = inputs[:, 3:]

        left_features = self.feature_extractor(left_images)
        right_features = self.feature_extractor(right_images)

        combined_features = torch.cat((left_features, right_features), dim=1)
        is_paired = self.is_paired_classifier(combined_features)
        is_swapped = self.is_swapped_classifier(combined_features)
        xy = self.xy_regressor(combined_features)

        return dict(outputs=xy, is_paired=is_paired, is_swapped=is_swapped)

    def set_alpha(self, value):
        self.alpha = float(value)

    def calculate_loss(self, targets, outputs, is_paired, is_swapped, **kwargs):
        paired_targets = targets[0].squeeze()
        swapped_targets = targets[1].squeeze()
        xy_targets = targets[2]

        paired_loss = F.nll_loss(input=is_paired, target=paired_targets)

        mask = paired_targets.float()
        s = F.nll_loss(input=is_swapped, target=swapped_targets, reduce=False) * mask
        swapped_loss = torch.sum(s) / torch.sum(mask)

        s = (outputs - xy_targets).pow(2) * mask.unsqueeze(dim=1)
        xy_loss = torch.sum(torch.sum(s, dim=1)) / torch.sum(mask)

        total_loss = (1.0 - self.alpha) * (paired_loss + swapped_loss) + (self.alpha) * xy_loss
        return dict(loss=total_loss, paired_loss=paired_loss, swapped_loss=swapped_loss, xy_loss=xy_loss)

    def calculate_metrics(self, targets, outputs, is_paired, is_swapped, **kwargs):
        paired_out = np.argmax(is_paired.cpu().data.numpy(), axis=1)
        paired_targets = targets[0].squeeze()
        paired_in = paired_targets.cpu().data.numpy()
        num_correct = (paired_out - paired_in == 0).sum()
        paired_accuracy = num_correct / len(paired_in)


        swapped_targets = targets[1].squeeze()
        swapped_in = swapped_targets.cpu().data.numpy()

        swapped_out = np.argmax(is_swapped.cpu().data.numpy(), axis=1)
        mask = paired_in
        swapped_out_masked = np.where(mask, swapped_out, 2*np.ones(swapped_out.shape))

        swapped_num_correct = (swapped_in - swapped_out_masked == 0).sum()
        swapped_accuracy = swapped_num_correct / mask.sum()

        return dict(paired_accuracy=paired_accuracy, swapped_accuracy=swapped_accuracy)
