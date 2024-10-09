import numpy as np
import torch
import torch.nn as nn
from pcdet.models.backbones_3d.dsvt_input_layer import DSVTInputLayer
from torch.utils.checkpoint import checkpoint


class LightDecoder(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.mask_token = nn.Parameter(torch.zeros(1, self.model_cfg.get('d_model', 128)[0]))
        self.input_layer = DSVTInputLayer(self.model_cfg.INPUT_LAYER)
        # save GPU memory
        self.use_torch_ckpt = self.model_cfg.get('ues_checkpoint', False)

        set_info = self.model_cfg.set_info
        block_name = self.model_cfg.block_name
        d_model = self.model_cfg.d_model
        nhead = self.model_cfg.nhead
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation

        # Sparse Regional Attention Blocks
        stage_num = len(block_name)
        for stage_id in range(stage_num):
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = d_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            block_name_this_stage = block_name[stage_id]
            block_module = _get_block_module(block_name_this_stage)
            block_list=[]
            norm_list=[]
            for i in range(num_blocks_this_stage):
                block_list.append(
                    block_module(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                                 dropout, activation, batch_first=True)
                )
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list))

            # apply pooling except the last stage
            if stage_id < stage_num-1:
                downsample_window = self.model_cfg.INPUT_LAYER.downsample_stride[stage_id]
                dmodel_next_stage = d_model[stage_id+1]
                pool_volume = torch.IntTensor(downsample_window).prod().item()
                if self.reduction_type == 'linear':
                    cat_feat_dim = dmodel_this_stage * torch.IntTensor(downsample_window).prod().item()
                    self.__setattr__(f'stage_{stage_id}_reduction', Stage_Reduction_Block(cat_feat_dim, dmodel_next_stage))
                elif self.reduction_type == 'maxpool':
                    self.__setattr__(f'stage_{stage_id}_reduction', torch.nn.MaxPool1d(pool_volume))
                elif self.reduction_type == 'attention':
                    self.__setattr__(f'stage_{stage_id}_reduction', Stage_ReductionAtt_Block(dmodel_this_stage, pool_volume))
                else:
                    raise NotImplementedError
        self.num_shifts = [2] * stage_num
        self.stage_num = stage_num
        self.set_info = set_info

        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        coors_mask = data_dict['voxel_coords_bk'][data_dict['voxel_mae_mask'].bool()]
        masked_start_id = data_dict['voxel_coords'].shape[0]
        mask_tokens = self.mask_token.repeat(coors_mask.shape[0], 1)
        voxel_feat_ = torch.cat([data_dict['voxel_features'], mask_tokens], dim=0)
        coors_ = torch.cat([data_dict['voxel_coords'], coors_mask], dim=0)

        batch_dict = {
            'voxel_features': voxel_feat_,
            'voxel_coords': coors_
        }

        voxel_info = self.input_layer(batch_dict)

        voxel_feat = voxel_info['voxel_feats_stage0']
        set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for
                               s in range(self.stage_num)]
        set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for
                                s in range(self.stage_num)]
        pos_embed_list = [
            [[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}'] for i in range(self.num_shifts[s])] for b in
             range(self.set_info[s][1])] for s in range(self.stage_num)]
        pooling_mapping_index = [voxel_info[f'pooling_mapping_index_stage{s + 1}'] for s in range(self.stage_num - 1)]
        pooling_index_in_pool = [voxel_info[f'pooling_index_in_pool_stage{s + 1}'] for s in range(self.stage_num - 1)]
        pooling_preholder_feats = [voxel_info[f'pooling_preholder_feats_stage{s + 1}'] for s in
                                   range(self.stage_num - 1)]

        output = voxel_feat
        block_id = 0
        for stage_id in range(self.stage_num):
            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(f'residual_norm_stage_{stage_id}')
            for i in range(len(block_layers)):
                block = block_layers[i]
                residual = output.clone()
                if self.use_torch_ckpt == False:
                    output = block(output, set_voxel_inds_list[stage_id], set_voxel_masks_list[stage_id],
                                   pos_embed_list[stage_id][i], \
                                   block_id=block_id)
                else:
                    output = checkpoint(block, output, set_voxel_inds_list[stage_id], set_voxel_masks_list[stage_id],
                                        pos_embed_list[stage_id][i], block_id)
                output = residual_norm_layers[i](output + residual)
                block_id += 1
            if stage_id < self.stage_num - 1:
                # pooling
                prepool_features = pooling_preholder_feats[stage_id].type_as(output)
                pooled_voxel_num = prepool_features.shape[0]
                pool_volume = prepool_features.shape[1]
                prepool_features[pooling_mapping_index[stage_id], pooling_index_in_pool[stage_id]] = output
                prepool_features = prepool_features.view(prepool_features.shape[0], -1)

                if self.reduction_type == 'linear':
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features)
                elif self.reduction_type == 'maxpool':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features).squeeze(-1)
                elif self.reduction_type == 'attention':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    key_padding_mask = torch.zeros((pooled_voxel_num, pool_volume)).to(prepool_features.device).int()
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features, key_padding_mask)
                else:
                    raise NotImplementedError

        data_dict['pillar_features_decoder'] = data_dict['voxel_features_decoder'] = output
        batch_dict['voxel_coords_bk'] = voxel_info[f'voxel_coors_stage{self.stage_num - 1}']

        # PointPillarScatter3d
        features = output
        coords = voxel_info[f'voxel_coors_stage{self.stage_num - 1}']

        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=features.dtype,
                device=features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size,
                                                             self.num_bev_features_before_compression * self.nz,
                                                             self.ny, self.nx)
        data_dict['spatial_features'] = batch_spatial_features

        return data_dict


class Stage_Reduction_Block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.linear1 = nn.Linear(input_channel, output_channel, bias=False)
        self.norm = nn.LayerNorm(output_channel)

    def forward(self, x):
        src = x
        src = self.norm(self.linear1(x))
        return src

class Stage_ReductionAtt_Block(nn.Module):
    def __init__(self, input_channel, pool_volume):
        super().__init__()
        self.pool_volume = pool_volume
        self.query_func = torch.nn.MaxPool1d(pool_volume)
        self.norm = nn.LayerNorm(input_channel)
        self.self_attn = nn.MultiheadAttention(input_channel, 8, batch_first=True)
        self.pos_embedding = nn.Parameter(torch.randn(pool_volume, input_channel))
        nn.init.normal_(self.pos_embedding, std=.01)

    def forward(self, x, key_padding_mask):
        # x: [voxel_num, c_dim, pool_volume]
        src = self.query_func(x).permute(0, 2, 1)  # voxel_num, 1, c_dim
        key = value = x.permute(0, 2, 1)
        key = key + self.pos_embedding.unsqueeze(0).repeat(src.shape[0], 1, 1)
        query = src.clone()
        output = self.self_attn(query, key, value, key_padding_mask)[0]
        src = self.norm(output + src).squeeze(1)
        return src

def _get_block_module(name):
    """Return an block module given a string"""
    if name == "DSVTBlock":
        return DSVTBlock
    raise RuntimeError(F"This Block not exist.")

class DSVTBlock(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True):
        super().__init__()

        encoder_1 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first)
        encoder_2 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(
            self,
            src,
            set_voxel_inds_list,
            set_voxel_masks_list,
            pos_embed_list,
            block_id,
    ):
        num_shifts = 2
        output = src
        # TODO: bug to be fixed, mismatch of pos_embed
        for i in range(num_shifts):
            set_id = i
            shift_id = block_id % 2
            pos_embed_id = i
            set_voxel_inds = set_voxel_inds_list[shift_id][set_id]
            set_voxel_masks = set_voxel_masks_list[shift_id][set_id]
            pos_embed = pos_embed_list[pos_embed_id]
            layer = self.encoder_list[i]
            output = layer(output, set_voxel_inds, set_voxel_masks, pos_embed)

        return output

class DSVT_EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, mlp_dropout=0):
        super().__init__()
        self.win_attn = SetAttention(d_model, nhead, dropout, dim_feedforward, activation, batch_first, mlp_dropout)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self,src,set_voxel_inds,set_voxel_masks,pos=None,onnx_export=False):
        identity = src
        src = self.win_attn(src, pos, set_voxel_masks, set_voxel_inds, onnx_export)
        src = src + identity
        src = self.norm(src)

        return src

class SetAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout, dim_feedforward=2048, activation="relu", batch_first=True, mlp_dropout=0):
        super().__init__()
        self.nhead = nhead
        if batch_first:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos=None, key_padding_mask=None, voxel_inds=None, onnx_export=False):
        '''
        Args:
            src (Tensor[float]): Voxel features with shape (N, C), where N is the number of voxels.
            pos (Tensor[float]): Position embedding vectors with shape (N, C).
            key_padding_mask (Tensor[bool]): Mask for redundant voxels within set. Shape of (set_num, set_size).
            voxel_inds (Tensor[int]): Voxel indexs for each set. Shape of (set_num, set_size).
            onnx_export (bool): Substitute torch.unique op, which is not supported by tensorrt.
        Returns:
            src (Tensor[float]): Voxel features.
        '''
        set_features = src[voxel_inds]
        if pos is not None:
            set_pos = pos[voxel_inds]
        else:
            set_pos = None
        if pos is not None:
            query = set_features + set_pos
            key = set_features + set_pos
            value = set_features

        if key_padding_mask is not None:
            src2 = self.self_attn(query, key, value, key_padding_mask)[0]
        else:
            src2 = self.self_attn(query, key, value)[0]

        # map voxel featurs from set space to voxel space: (set_num, set_size, C) --> (N, C)
        flatten_inds = voxel_inds.reshape(-1)
        if onnx_export:
            src2_placeholder = torch.zeros_like(src).to(src2.dtype)
            src2_placeholder[flatten_inds] = src2.reshape(-1, self.d_model)
            src2 = src2_placeholder
        else:
            unique_flatten_inds, inverse = torch.unique(flatten_inds, return_inverse=True)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(unique_flatten_inds.size(0)).scatter_(0, inverse, perm)
            src2 = src2.reshape(-1, self.d_model)[perm]

        # FFN layer
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class SSTBEVBackbone(nn.Module):
    def __init__(self, model_cfg, num_filters):
        super().__init__()
        self.model_cfg = model_cfg
        input_channels = num_filters

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.out_channels_map = nn.Sequential(
                nn.Conv2d(c_in, input_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        self.num_bev_features = input_channels

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        x = self.out_channels_map(x)
        data_dict['spatial_features'] = x

        return data_dict

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")