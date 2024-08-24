import torch.nn as nn
from ...ops.sst_ops import sst_ops_utils
from ...utils import common_utils
from pytorch3d.loss import chamfer_distance
import torch_scatter
import MinkowskiEngine as ME
import numpy as np
import torch
import os
import math
import random
from pcdet.models.dense_heads.target_assigner.res16unet import Res16UNet34C as MinkUNet
from pcdet.models import build_network
from pcdet.datasets import build_dataloader

class PretrainHead3D(nn.Module):
    def __init__(self, model_cfg, cfg, input_channels, layer_num, cur_epoch, class_names, voxel_size, point_cloud_range, grid_size, **kwargs):
        super().__init__()
        self.mask_cfg = model_cfg.get('MASK_CONFIG', None)
        self.model_cfg = model_cfg
        self.cfg = cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_seal_features = self.model_cfg.NUM_SEAL_FEATURES
        self.cka_alhpa = self.model_cfg.CKA_ALHPA
        self.gamma = self.model_cfg.GAMMA
        self.delta = self.model_cfg.DELTA
        self.generate_mode = model_cfg.get('GENERATE_MODE', 'offline')
        self.num_seal_features_before_compression = self.model_cfg.NUM_SEAL_FEATURES // self.nz
        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        self.input_channels = input_channels
        self.layer_num = layer_num
        self.cur_epoch = cur_epoch
        self.beta_t = self.differential_gated_progressive_learning(self.cur_epoch)
        self.class_names = class_names
        self.decoder_pred = nn.Linear(input_channels, self.mask_cfg.NUM_PRD_POINTS * 3, bias=True)
        self.decoder_seal = nn.Linear(input_channels, self.num_seal_features, bias=True)
        self.seal_loss = nn.SmoothL1Loss(reduction='none')

    def target_assigner(self, batch_dict):
        # sparse_input = ME.SparseTensor(batch_dict['unique_feats'], batch_dict['discrete_coords'])
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

        # MinkUNet (Res16UNet34C)
        if self.generate_mode == 'online':
            # online generate seal features
            feats = batch_dict['unique_feats'][:, :1]
            coords = batch_dict['discrete_coords'].to(torch.int32)

            load_path = '/path/of/the/seal/model.pt'

            model_points = MinkUNet(1, me_cfg["model_n_out"], me_cfg)
            # load pretrain model
            checkpoint = torch.load(load_path, map_location="cpu")
            key = "state_dict"
            filtered_weights = load_state_with_same_shape(model_points, checkpoint[key])
            model_dict = model_points.state_dict()
            model_dict.update(filtered_weights)
            model_points.load_state_dict(model_dict)

            model_points = model_points.to('cuda')
            model_points.eval()

            sparse_input = ME.SparseTensor(feats, coords)
            output_points = model_points(sparse_input).F

        elif self.generate_mode == 'offline':
            # offline load seal features
            seal_outputs_dir = '/path/of/seal/feature/'
            output_points = []
            for id in batch_dict['frame_id']:
                seal_output_path = seal_outputs_dir + id
                output_point = np.load(seal_output_path + '.npy')
                output_points.append(output_point)
            output_points = np.concatenate(output_points, axis=0)

        output_points = torch.tensor(output_points).cuda()

        # get discrete point coords
        ori_coords = batch_dict['ori_coords']
        indexes = batch_dict['indexes']
        discrete_coords = []
        for i in range(batch_size):
            index = indexes[indexes[:, 0].to(torch.int64) == i][:, 1].to(torch.int64)
            ori_coord = ori_coords[ori_coords[:, 0].to(torch.int64) == i]
            discrete_coord = ori_coord[index]
            discrete_coords.append(discrete_coord)
        discrete_coords = torch.cat(discrete_coords, dim=0)

        # DynPillarVFE3D
        points_coords = torch.floor(
            (discrete_coords[:, [1, 2, 3]] - self.point_cloud_range[[0, 1, 2]]) / self.voxel_size[[0, 1, 2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1, 2]])).all(dim=1)
        discrete_coords = discrete_coords[mask]
        points_coords = points_coords[mask]
        output_points = output_points[mask]

        merge_coords = discrete_coords[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        output_points_mean = torch_scatter.scatter_mean(output_points, unq_inv, dim=0)

        target_voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        target_voxel_coords = target_voxel_coords[:, [0, 3, 2, 1]]

        # PointPillarScatter3d
        batch_spatial_features = []
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_seal_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=output_points_mean.dtype,
                device=output_points_mean.device)

            batch_mask = target_voxel_coords[:, 0] == batch_idx
            this_coords = target_voxel_coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = output_points_mean[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_seal_features_before_compression * self.nz,
                                                             self.ny, self.nx)

        # prepare high-level targets and preds
        ori_voxel_coords = batch_dict['voxel_coords_bk']
        slices = [ori_voxel_coords[:, i].long() for i in [0, 2, 3]]
        target_seal_voxel_features = batch_spatial_features.permute(0, 2, 3, 1)[slices]

        pred_seal_voxel_features = self.decoder_seal(all_voxel_features).view(all_voxel_features.shape[0], -1)
        non_empty_seal_mask = ~torch.all(target_seal_voxel_features == 0, dim=1)

        target_seal_voxel_features = target_seal_voxel_features[non_empty_seal_mask]
        pred_seal_voxel_features = pred_seal_voxel_features[non_empty_seal_mask]
        seal_mae_mask = voxel_mae_mask[non_empty_seal_mask]

        forward_ret_dict['pred_seal_voxel_features'] = pred_seal_voxel_features
        forward_ret_dict['target_seal_voxel_features'] = target_seal_voxel_features
        forward_ret_dict['seal_mae_mask'] = seal_mae_mask


        return forward_ret_dict

    def differential_gated_progressive_learning(self, cur_epoch):
        if cur_epoch < 2:
            return 0
        
        # Initialize two models
        model_t_1 = build_network(model_cfg=self.model_cfg)
        model_t_2 = build_network(model_cfg=self.model_cfg)

        weights_dir = 'path/to/weights'

        # Load weights
        weights_t_1_path = os.path.join(weights_dir, f'epoch_{cur_epoch-1}.pth')
        weights_t_2_path = os.path.join(weights_dir, f'epoch_{cur_epoch-2}.pth')

        model_t_1 = model_t_1.load_state_dict(torch.load(weights_t_1_path))
        model_t_2 = model_t_2.load_state_dict(torch.load(weights_t_2_path))

        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=self.cfg.DATA_CONFIG,
            class_names=self.cfg.CLASS_NAMES,
            batch_size=1,
            training=False
        )
        test_sample = test_loader[random.randint(0, len(test_loader))]

        output_batch_dict_t_1 = model_t_1(test_sample)
        output_batch_dict_t_2 = model_t_2(test_sample)

        # Calculate the CKA of the first 4 layers
        CKAs = []
        for i in range(4):
            activation_t_1 = output_batch_dict_t_1[f'block_layer_{i}']
            activation_t_2 = output_batch_dict_t_2[f'block_layer_{i}']
            CKA = self.linear_cka(activation_t_1, activation_t_2)
            CKAs.append(CKA)

        # Calculate the average CKA
        c_r = sum(CKAs) / len(CKAs)

        if 0 <= c_r < self.delta:
            return 0
        elif self.delta <= c_r <= 1:
            return 1 - math.exp(-self.gamma * (c_r - self.delta))
        else:
            raise ValueError("crt should be in the range [0, 1]")
        
    def linear_cka(X, Y):
        # Ensure the data is centered (centering operation)
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)

        # Compute the Gram matrices
        K = np.dot(X, X.T)
        L = np.dot(Y, Y.T)

        # Compute the Frobenius norms
        norm_K = np.linalg.norm(K, 'fro')
        norm_L = np.linalg.norm(L, 'fro')

        # Calculate Linear CKA
        cka = np.dot(K.flatten(), L.flatten()) / (norm_K * norm_L)
    
        return cka
    
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        loss = 0
        # (N, K, 3)
        gt_points, pred_points, mask = \
            self.forward_ret_dict['gt_points'], self.forward_ret_dict['pred_points'], self.forward_ret_dict['mask']
        loss_points, _ = chamfer_distance(pred_points, gt_points, weights=mask)
        loss += loss_points * self.cka_alhpa[self.layer_num]
        tb_dict['loss_points'] = loss_points.item()
        # loss sealowskiengine
        pred_seal_voxel_features, target_seal_voxel_features, seal_mae_mask = \
            self.forward_ret_dict['pred_seal_voxel_features'], self.forward_ret_dict['target_seal_voxel_features'], self.forward_ret_dict['seal_mae_mask']
        # pred_seal_voxel_features = F.normalize(pred_seal_voxel_features, dim=1)
        # target_seal_voxel_features = F.normalize(target_seal_voxel_features, dim=1)
        loss_seal = self.seal_loss(pred_seal_voxel_features, target_seal_voxel_features)
        loss_seal = loss_seal[seal_mae_mask.bool()]
        loss_seal = loss_seal.mean()
        loss += loss_seal * (1 - self.cka_alhpa[self.layer_num]) * self.beta_t
        tb_dict['loss_seal'] = loss_seal.item()
        return loss, tb_dict

    def forward(self, batch_dict):
        all_voxel_coords = batch_dict['voxel_coords_bk']
        slices = [all_voxel_coords[:, i].long() for i in [0, 2, 3]]
        all_voxel_features = batch_dict['spatial_features'].permute(0, 2, 3, 1)[slices]

        batch_dict['all_voxel_features'] = all_voxel_features
        batch_dict['all_voxel_coords'] = all_voxel_coords

        self.forward_ret_dict = self.target_assigner(batch_dict)

        return batch_dict

def load_state_with_same_shape(model, weights):
    """
    Load common weights in two similar models
    (for instance between a pretraining and a downstream training)
    """
    model_state = model.state_dict()
    if list(weights.keys())[0].startswith("model."):
        weights = {k.partition("model.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("model_points."):
        weights = {k.partition("model_points.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("module."):
        print("Loading multigpu weights with module. prefix...")
        weights = {k.partition("module.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("encoder."):
        print("Loading multigpu weights with encoder. prefix...")
        weights = {k.partition("encoder.")[2]: weights[k] for k in weights.keys()}

    filtered_weights = {
        k: v
        for k, v in weights.items()
        if (k in model_state and v.size() == model_state[k].size())
    }
    removed_weights = {
        k: v
        for k, v in weights.items()
        if not (k in model_state and v.size() == model_state[k].size())
    }
    print("Loading weights:" + ", ".join(filtered_weights.keys()))
    print("")
    print("Not loading weights:" + ", ".join(removed_weights.keys()))
    return filtered_weights

me_cfg = {
    'kernel_size': 3,
    'normalize_features': True,
    'model_n_out': 64,
    'bn_momentum': 0.05
}