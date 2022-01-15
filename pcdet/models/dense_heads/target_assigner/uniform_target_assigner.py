import torch
import numpy as np

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import common_utils
from ....utils import box_utils


class UniformMatchTargetAssigner(object):
    def __init__(self, topk, model_cfg,  box_coder, class_names, match_height=False):
        self.topk = topk
        self.match_height = match_height
        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.neg_ignore_thr = {}
        self.pos_ignore_thr = {}
        for config in anchor_generator_cfg:
            self.neg_ignore_thr[config['class_name']] = config['neg_ignore_threshold']
            self.pos_ignore_thr[config['class_name']] = config['pos_ignore_threshold']

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)
    
    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: [(N, 7), ...] 这是一个列表，分别存储每一类的生成的anchor。
            对于all_anchor中的每一个元素，其大小为[1, 200, 176,  1, 2, 7]。
            前三个数应该是feature map的大小，其中1应该是1个head的意思
            后三个数中的2应该是表示的两个方向的anchor,7表示的是anchor的大小
            1应该是表示的是head的数目
            gt_boxes: (B, M, 8)
        Returns:

        """

        bbox_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        # 一个batch一个batch的处理
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            # 获取真实的GT的数目
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            target_list = []
            # 需要逐个处理不同的类别
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                # 下面获得的mask对应的是和当前处理的类别一致的GT值
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3]
                    anchors = anchors.view(-1, anchors.shape[-1])
                    selected_classes = cur_gt_classes[mask]

                single_target = self.assign_targets_single(
                    anchors,
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    neg_ignore_thr=self.neg_ignore_thr[anchor_class_name],
                    pos_ignore_thr=self.pos_ignore_thr[anchor_class_name]
                )
                target_list.append(single_target)

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]
                }
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)

                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)

            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])

        bbox_targets = torch.stack(bbox_targets, dim=0)

        cls_labels = torch.stack(cls_labels, dim=0)
        reg_weights = torch.stack(reg_weights, dim=0)
        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights

        }
        return all_targets_dict
    
    # 不同类别需要设置不同的阈值，因为不同类别的大小不一样
    def assign_targets_single(self, anchors, gt_boxes, gt_classes, neg_ignore_thr=0.6, pos_ignore_thr=0.45):
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        labels = anchors.new_zeros((num_anchors,), dtype=torch.int32) # 这个应该标注的是这个anchor是什么类别 [-1 表示忽略, 0表示背景，其余为其他类别]
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1  # 这个应该表示的是这个anchor对应的GT的下标


        # 如果需要进行绑定的话
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # 计算anchor和GT之间的下标，如果使用height的话就计算三维的IoU,如果不match height的话，就计算BEV IoU
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])
            
            # +1 防止出现除0的问题
            anchor_by_gt_distance = 1.0 / ((anchors[:, None, 0:3] - gt_boxes[None, :, 0:3]).norm(dim=-1) + 1)  # (num_anchors, num_gt)
            # 首先筛选出IoU > 0.7
            anchor_to_gt_overlap_max = torch.max(anchor_by_gt_overlap, dim=1)[0]
            tmp_ignore_idx = anchor_to_gt_overlap_max > neg_ignore_thr
            labels[tmp_ignore_idx] = -1


            # anchor_by_gt_metric = anchor_by_gt_overlap * anchor_by_gt_distance
            anchor_by_gt_metric = anchor_by_gt_distance
            # 找到关于每一个GT的具有最大metric的anchor的下标
            topk_values, topk_idxs = anchor_by_gt_metric.topk(k=self.topk, dim=0, largest=True) #[k, num_gt]
            # 将topk_values的也设置为忽略
            
            candidate_ious = anchor_by_gt_overlap[topk_idxs, torch.arange(num_gt)]  # (K, num_gt)
            # print(candidate_ious)

            pos_ious_ignore_idxs = candidate_ious < pos_ignore_thr
            pos_ignore_idxs = topk_idxs[pos_ious_ignore_idxs]
            labels[pos_ignore_idxs] = -1

            pos_ious_idxs = candidate_ious >= pos_ignore_thr
            pos_gt_idxs = pos_ious_idxs.nonzero()[:, 1]
            # 如果获得这个pos的对应的anchor的下标
            
            pos_anchor_idxs = topk_idxs[pos_ious_idxs]
            if len(pos_gt_idxs) > 0 and len(pos_anchor_idxs) > 0:
                assert len(pos_gt_idxs) == len(pos_anchor_idxs), "维度不对"
                labels[pos_anchor_idxs] = gt_classes[pos_gt_idxs]
                gt_ids[pos_anchor_idxs] = pos_gt_idxs.int()
            else:
                print("没有正样本")
        
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[pos_gt_idxs]
            fg_anchors = anchors[pos_anchor_idxs]
            bbox_targets[pos_anchor_idxs, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        reg_weights = anchors.new_zeros((num_anchors,))

        # 是否根据前景点的数量来改变不同anchor的权重
        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum() # 这里是获得划分为positive的anchor的数量
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
        }
        return ret_dict

            
