from select import select
import numpy as np
import torch.nn as nn
import torch

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadDis(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.conv_clss = []
        self.conv_boxs = []
        self.conv_dir_clss = []

        self.dis_bin = self.model_cfg.LENGTH_BIN

        self.conv_clss = [nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )] * self.dis_bin

        self.conv_boxs = [nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )] * self.dis_bin

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_clss = [nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )] * self.dis_bin
            self.conv_dir_clss = nn.ModuleList(self.conv_dir_clss)
        else:
            self.conv_dir_clss = None

        self.conv_clss = nn.ModuleList(self.conv_clss)
        self.conv_boxs = nn.ModuleList(self.conv_boxs)
        
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        for conv_cls, conv_box in zip(self.conv_clss, self.conv_boxs):
            nn.init.constant_(conv_cls.bias, -np.log((1 - pi) / pi))
            nn.init.normal_(conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = []
        box_preds = []
        dir_cls_preds = []
        
        length = int(spatial_features_2d.shape[-1] / self.dis_bin)
        for i in range(self.dis_bin):
            if i == self.dis_bin - 1:
                feature_2d = spatial_features_2d[:,:,(i * length):,]
            else:
                feature_2d = spatial_features_2d[:, :, (i * length) : ((i + 1)* length),]
            
            cls_preds.append(self.conv_clss[i](feature_2d))
            box_preds.append(self.conv_boxs[i](feature_2d))
            
            if self.conv_dir_clss is not None:
                dir_cls_preds.append(self.conv_dir_clss[i](feature_2d))
        
        cls_preds = torch.cat(cls_preds, dim=-2)
        box_preds = torch.cat(box_preds, dim=-2)


        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, num_anchors*num_class]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_clss is not None:
            dir_cls_preds = torch.cat(dir_cls_preds, dim=-2)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
