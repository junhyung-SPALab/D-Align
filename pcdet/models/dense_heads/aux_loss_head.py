import copy
import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils

class Heatmap_Similarity_Loss_Head(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 sequence_length):
        super().__init__()
        self.model_cfg = model_cfg
        self.module_category = self.model_cfg.MODULE_CATEGORY
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.embedding_network = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.EMBEDDING_OUTPUT_CHANNEL, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.model_cfg.EMBEDDING_OUTPUT_CHANNEL),
            nn.ReLU(),
        )
        
        for module in self.embedding_network.modules():
            if isinstance(module, nn.Conv2d):
                kaiming_normal_(module.weight.data)        

        self.num_class = self.model_cfg.NUM_TOTAL_CLASS
        self.seq_length = sequence_length

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def heatmap_similarity_loss_per_single_head(self, L2_dist_ret, heatmap):
        """
        Args:
            pred: (batch x c x h x w)
            gt: (batch x c x h x w)
            mask: (batch x h x w)
        Returns:
        """
        num_loc = heatmap.gt(0).float().sum()
        if num_loc == 0:
            return 0
        else:
            heatmap_sim_loss = L2_dist_ret * heatmap
            heatmap_sim_loss = heatmap_sim_loss.sum()
            return (heatmap_sim_loss / num_loc)

    

    def get_loss(self, embed_features_2d_list: list, heatmap_ref_dict: dict):
        loss = 0
        L2_dist_ret_list = []

        batch, C, H, W = embed_features_2d_list[-1].size()
        zero_vector = torch.zeros(batch, 1, C).to(embed_features_2d_list[-1].device)

        for t in range(self.seq_length-1):
            support_feature_norm_out    = F.normalize(embed_features_2d_list[t], dim=1)
            ref_feature_norm_out        = F.normalize(embed_features_2d_list[-1], dim=1)
            residual = (ref_feature_norm_out-support_feature_norm_out).view(batch,C,-1).transpose(1,2)
            # residual = (embed_features_2d_list[-1]-embed_features_2d_list[t]).view(batch,C,-1).transpose(1,2)
            dist = torch.cdist(residual, zero_vector).transpose(1,2).view(batch, 1, H, W)
            L2_dist_ret_list.append(dist)
        assert len(L2_dist_ret_list) == self.seq_length-1

        loss_denominator = [self.num_class * (self.seq_length-1)] * batch
        loss_per_batch_list = [0] * batch

        for t in range(self.seq_length-1):
            for idx, heatmap_ref in enumerate(heatmap_ref_dict['heatmaps']):
                for batch_idx in range(batch):
                    heatmap_similarity_loss = self.heatmap_similarity_loss_per_single_head(L2_dist_ret_list[t][batch_idx], heatmap_ref[batch_idx])
                    if heatmap_similarity_loss == 0:
                        loss_denominator[batch_idx] -= heatmap_ref[batch_idx].size()[1]
                        continue
                    loss_per_batch_list[batch_idx] += heatmap_similarity_loss
        final_loss_per_batch_list = [loss_per_batch_list[batch_idx] / loss_denominator[batch_idx] for batch_idx in range(batch)]
        assert len(final_loss_per_batch_list) == batch
        final_loss = sum(final_loss_per_batch_list) / batch

        return final_loss


    def forward(self, data_dict_seq_list):
        embed_features_2d_list = []
        for t in range(self.seq_length):
            if t == (self.seq_length-1):
                embed_features_2d_list.append(self.embedding_network(data_dict_seq_list[t]['spatial_features_2d']))
            else:
                embed_features_2d_list.append(self.embedding_network(data_dict_seq_list[t]['aligned_features_2d']))
        assert len(embed_features_2d_list) == self.seq_length

        target_data_dict = data_dict_seq_list[-1]
        
        
        heatmap_ref_dict = self.assign_targets(
            target_data_dict['gt_boxes'], feature_map_size=embed_features_2d_list[-1].size()[2:],
            feature_map_stride=target_data_dict.get('spatial_features_2d_strides', None)
        )
        
        similarity_loss = self.get_loss(embed_features_2d_list, heatmap_ref_dict)
        similarity_loss = similarity_loss * self.model_cfg.AUX_LOSS_WEIGHTS
        return similarity_loss