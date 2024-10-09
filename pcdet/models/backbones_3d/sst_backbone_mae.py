import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from I2Mask import I2Mask_func
import torch_scatter
from ...utils.spconv_utils import spconv
from .sst_backbone import SSTBlockV1
from ...utils import common_utils
from ...ops.sst_ops import sst_ops_utils
from pytorch3d.loss import chamfer_distance
from pcdet.models.dense_heads.target_assigner.res16unet import Res16UNet34C as MinkUNet


class SSTBackboneMAE(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = self.model_cfg.grid_size
        self.voxel_size = self.model_cfg.voxel_size
        self.point_cloud_range = self.model_cfg.point_cloud_range
        self.sparse_shape = self.model_cfg.grid_size[[1, 0]]
        in_channels = self.model_cfg.input_channels
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_LAYER.sparse_shape

        self.mask_cfg = self.model_cfg.get('MASK_CONFIG', None)
        self.mask_func = I2Mask_func(self.mask_cfg.get('n_clusters', 8), self.mask_cfg.get('n_partition', [3, 3, 2]), self.mask_cfg.get('lambda_threshold', 0.6),
                                     self.mask_cfg.get('base_mask_ratio', [0.9, 0.45, 0]))
        self.generate_mode = self.model_cfg.get('GENERATE_MODE', None)

        self.num_seal_features = self.model_cfg.NUM_SEAL_FEATURES
        self.num_seal_features_before_compression = self.model_cfg.NUM_SEAL_FEATURES // self.nz

        sst_block_list = model_cfg.SST_BLOCK_LIST
        self.sst_blocks = nn.ModuleList()
        for sst_block_cfg in sst_block_list:
            self.sst_blocks.append(SSTBlockV1(sst_block_cfg, in_channels, sst_block_cfg.NAME))
            in_channels = sst_block_cfg.ENCODER.D_MODEL
        
        in_channels = 0
        self.decoder_deblocks = nn.ModuleList()
        for src in model_cfg.FEATURES_SOURCE:
            conv_cfg = model_cfg.FUSE_LAYER[src]
            self.decoder_deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    conv_cfg.NUM_FILTER, conv_cfg.NUM_UPSAMPLE_FILTER,
                    conv_cfg.UPSAMPLE_STRIDE,
                    stride=conv_cfg.UPSAMPLE_STRIDE, bias=False
                ),
                nn.BatchNorm2d(conv_cfg.NUM_UPSAMPLE_FILTER, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ))
            in_channels += conv_cfg.NUM_UPSAMPLE_FILTER
        
        self.decoder_conv_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // len(self.decoder_deblocks), 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // len(self.decoder_deblocks), eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        in_channels = in_channels // len(self.decoder_deblocks)

        self.decoder_pred = nn.Linear(in_channels, self.mask_cfg.NUM_PRD_POINTS * 3, bias=True)
        self.forward_ret_dict = {}

        self.num_point_features = in_channels

    def target_assigner(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        voxel_shuffle_inds = batch_dict['voxel_shuffle_inds']
        points = batch_dict['points']
        point_inverse_indices = batch_dict['point_inverse_indices']
        voxel_mae_mask = batch_dict['voxel_mae_mask']
        # road_plane = batch_dict['road_plane']
        batch_size = batch_dict['batch_size']

        gt_points = sst_ops_utils.group_inner_inds(points[:, 1:4], point_inverse_indices, self.mask_cfg.NUM_GT_POINTS)
        gt_points = gt_points[voxel_shuffle_inds]
        voxel_centers = common_utils.get_voxel_centers(
            voxel_coords[:, 1:], 1, self.voxel_size, self.point_cloud_range, dim=3
        )  # (N, 3)
        norm_gt_points = gt_points - voxel_centers.unsqueeze(1)
        mask = voxel_mae_mask[voxel_shuffle_inds]
        pred_points = self.decoder_pred(voxel_features).view(voxel_features.shape[0], -1, 3)

        forward_ret_dict = {
            'pred_points': pred_points,  # (N, P1, 3)
            'gt_points': norm_gt_points,  # (N, P2, 3)
            'mask': mask  # (N,)
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
        all_voxel_features, all_voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        assert torch.all(all_voxel_coords[:, 1] == 0)

        # Prepare Seal feature for I2Mask
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

        voxel_mae_mask = []
        for bs_idx in range(batch_size):
            voxel_mae_mask.append(common_utils.random_masking(1, (all_voxel_coords[:, 0] == bs_idx).sum().item(), self.mask_ratio, all_voxel_coords.device)[0])
        voxel_mae_mask = torch.cat(voxel_mae_mask, dim=0)
        batch_dict['voxel_mae_mask'] = voxel_mae_mask

        
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
        batch_spatial_features = batch_spatial_features.view(batch_size,
                                                             self.num_seal_features_before_compression * self.nz,
                                                             self.ny, self.nx)
        
        # get seal feature
        ori_voxel_coords = batch_dict['voxel_coords']
        slices = [ori_voxel_coords[:, i].long() for i in [0, 2, 3]]
        seal_features = batch_spatial_features.permute(0, 2, 3, 1)[slices]

        voxel_mae_mask = []
        for bs_idx in range(batch_size):
            voxel_mae_mask.append(self.mask_func(seal_features[all_voxel_coords[:, 0] == bs_idx]))
        voxel_mae_mask = torch.cat(voxel_mae_mask, dim=0)
        batch_dict['voxel_mae_mask'] = voxel_mae_mask

        keep_voxel_features = all_voxel_features[voxel_mae_mask == 0]
        keel_voxel_coords = all_voxel_coords[voxel_mae_mask == 0].int()  # (bs_idx, z_idx, y_idx, x_idx)
        batch_dict['pillar_features_bk'] = batch_dict['pillar_features']
        batch_dict['voxel_features_bk'] = batch_dict['voxel_features']
        batch_dict['voxel_coords_bk'] = batch_dict['voxel_coords']
        batch_dict['voxel_features'] = keep_voxel_features
        batch_dict['voxel_coords'] = keel_voxel_coords

        input_sp_tensor = spconv.SparseConvTensor(
            features=all_voxel_features[voxel_mae_mask == 0],
            indices=all_voxel_coords[voxel_mae_mask == 0][:, [0, 2, 3]].contiguous().int(),  # (bs_idx, y_idx, x_idx)
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = input_sp_tensor
        x_hidden = []
        for sst_block in self.sst_blocks:
            x = sst_block(x)
            x_hidden.append(x)

        batch_dict.update({
            'encoded_spconv_tensor': x_hidden[-1],
            'encoded_spconv_tensor_stride': self.sparse_shape[0] // x_hidden[-1].spatial_shape[0]
        })

        multi_scale_3d_features, multi_scale_3d_strides = {}, {}
        for i in range(len(x_hidden)):
            multi_scale_3d_features[f'x_conv{i + 1}'] = x_hidden[i]
            multi_scale_3d_strides[f'x_conv{i + 1}'] = self.sparse_shape[0] // x_hidden[i].spatial_shape[0]
        
        spatial_features = []
        spatial_features_stride = []
        for i, src in enumerate(self.model_cfg.FEATURES_SOURCE):
            per_features = multi_scale_3d_features[src].dense()
            B, Y, X = per_features.shape[0], per_features.shape[-2], per_features.shape[-1]
            spatial_features.append(self.decoder_deblocks[i](per_features.view(B, -1, Y, X)))
            spatial_features_stride.append(multi_scale_3d_strides[src] // self.model_cfg.FUSE_LAYER[src].UPSAMPLE_STRIDE)
        spatial_features = self.decoder_conv_out(torch.cat(spatial_features, dim=1))  # (B, C, Y, X)
        spatial_features_stride = spatial_features_stride[0]
        
        batch_dict['multi_scale_3d_features'] = multi_scale_3d_features
        batch_dict['multi_scale_3d_strides'] = multi_scale_3d_strides
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = spatial_features_stride
        
        assert spatial_features.shape[0] == batch_size and spatial_features.shape[2] == self.grid_size[1] and spatial_features.shape[3] == self.grid_size[0]
        all_voxel_shuffle_inds = torch.arange(all_voxel_coords.shape[0], device=all_voxel_coords.device, dtype=torch.long)
        slices = [all_voxel_coords[:, i].long() for i in [0, 2, 3]]
        all_pyramid_voxel_features = spatial_features.permute(0, 2, 3, 1)[slices]

        target_dict = {
            'voxel_features': all_pyramid_voxel_features,
            'voxel_coords': all_voxel_coords,
            'voxel_shuffle_inds': all_voxel_shuffle_inds
        }
        batch_dict.update(target_dict)
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