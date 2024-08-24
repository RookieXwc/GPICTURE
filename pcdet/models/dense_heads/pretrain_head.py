import copy
import numpy as np
import torch
import torch.nn as nn
from ...ops.sst_ops import sst_ops_utils
from ...utils import common_utils
from pytorch3d.loss import chamfer_distance


class PretrainHead(nn.Module):
    def __init__(self, model_cfg, input_channels, class_names, voxel_size, point_cloud_range, grid_size, **kwargs):
        super().__init__()
        self.mask_cfg = model_cfg.get('MASK_CONFIG', None)
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.input_channels = input_channels
        self.class_names = class_names
        self.decoder_pred = nn.Linear(input_channels, self.mask_cfg.NUM_PRD_POINTS * 3, bias=True)

    def target_assigner(self, batch_dict):
        all_voxel_features = batch_dict['all_voxel_features']
        all_voxel_coords = batch_dict['all_voxel_coords']
        points = batch_dict['points']
        point_inverse_indices = batch_dict['point_inverse_indices']
        voxel_mae_mask = batch_dict['voxel_mae_mask']
        batch_size = batch_dict['batch_size']

        gt_points = sst_ops_utils.group_inner_inds(points[:, 1:4], point_inverse_indices, self.mask_cfg.NUM_GT_POINTS)
        voxel_centers = common_utils.get_voxel_centers(
            all_voxel_coords[:, 1:], 1, self.voxel_size, self.point_cloud_range
        )  # (N, 3)
        norm_gt_points = gt_points - voxel_centers.unsqueeze(1)
        pred_points = self.decoder_pred(all_voxel_features).view(all_voxel_features.shape[0], -1, 3)

        forward_ret_dict = {
            'pred_points': pred_points,  # (N, P1, 3)
            'gt_points': norm_gt_points,  # (N, P2, 3)
            'mask': voxel_mae_mask  # (N,)
        }


        return forward_ret_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # (N, K, 3)
        gt_points, pred_points, mask = \
            self.forward_ret_dict['gt_points'], self.forward_ret_dict['pred_points'], self.forward_ret_dict['mask']
        loss, _ = chamfer_distance(pred_points, gt_points, weights=mask)
        return loss, tb_dict

    def forward(self, batch_dict):
        all_voxel_coords = batch_dict['voxel_coords_bk']
        slices = [all_voxel_coords[:, i].long() for i in [0, 2, 3]]
        all_voxel_features = batch_dict['spatial_features'].permute(0, 2, 3, 1)[slices]

        batch_dict['all_voxel_features'] = all_voxel_features
        batch_dict['all_voxel_coords'] = all_voxel_coords

        self.forward_ret_dict = self.target_assigner(batch_dict)

        return batch_dict