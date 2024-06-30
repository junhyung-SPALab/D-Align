import copy
import enum
from tracemalloc import start
import numpy as np
from typing import List
from numpy.core.fromnumeric import shape

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_
from ...ops.deformDETR.modules import MSDeformAttn

class FPMNet(nn.Module):
    def __init__(self, in_channels:int, num_levels:int, scale_list:list):
        super(FPMNet, self).__init__()
        self.n_levels       = num_levels
        self.in_channels    = in_channels
        func_unflatten_dict = {}
        for lvl, scale_size in enumerate(scale_list):
            scale = 2**(lvl+1)
            func_unflatten_dict[f'unflatten_{scale}x'] = nn.Unflatten(dim=-1, unflattened_size=scale_size)
        self.func_unflatten_dict = nn.ModuleDict(func_unflatten_dict)

        if self.n_levels == 3:
            encode_channel      = int(self.in_channels//3)
            self.motion_encoder = nn.Conv2d(self.in_channels, encode_channel, kernel_size=3, stride=1, padding=1)

            self.block1_dec2x = nn.MaxPool2d(kernel_size=2)   ### C=64
            self.block1_dec4x = nn.MaxPool2d(kernel_size=4)   ### C=64

            self.block2_dec2x = nn.MaxPool2d(kernel_size=2)  ### C=128
            self.block2_inc2x = nn.ConvTranspose2d(encode_channel, encode_channel, kernel_size=2, stride=2)

            self.block3_inc2x = nn.ConvTranspose2d(encode_channel, encode_channel, kernel_size=2, stride=2)
            self.block3_inc4x = nn.ConvTranspose2d(encode_channel, encode_channel, kernel_size=4, stride=4)
        
        elif self.n_levels == 2:
            encode_channel      = int(self.in_channels//2)
            self.motion_encoder = nn.Conv2d(self.in_channels, encode_channel, kernel_size=3, stride=1, padding=1)
            self.block1_dec2x = nn.MaxPool2d(kernel_size=2)   ### C=64
            self.block2_inc2x = nn.ConvTranspose2d(encode_channel, encode_channel, kernel_size=2, stride=2)
        else:
            raise Exception()

        ms_motion_agg_list = [nn.Conv2d(encode_channel*self.n_levels, self.in_channels, kernel_size=1, stride=1, padding=0) for i in range(self.n_levels)]
        self.ms_motion_agg = torch.nn.ModuleList(ms_motion_agg_list)
        
        # Weight initialization
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    constant_(m.bias.data, 0)
    
    def forward(self, ref_spatial_src_MS:dict, target_spatial_src_MS:dict):
        ref_spatial_src_MS_2d       = [self.func_unflatten_dict[f'unflatten_{2**(lvl+1)}x'](torch.transpose(ref_spatial_src_MS[f'spatialAttn_result_{2**(lvl+1)}x'], 1,2)) for lvl in range(self.n_levels)]
        target_spatial_src_MS_2d    = [self.func_unflatten_dict[f'unflatten_{2**(lvl+1)}x'](torch.transpose(target_spatial_src_MS[f'temporalAttn_result_{2**(lvl+1)}x'], 1,2)) for lvl in range(self.n_levels)]
        
        motion_feature_list = []
        for lvl in range(self.n_levels):
            motion_feature = self.motion_encoder(ref_spatial_src_MS_2d[lvl] - target_spatial_src_MS_2d[lvl])
            motion_feature_list.append(motion_feature)

        fused_motion_feature_list = []
        
        if self.n_levels == 3:
            for lvl in range(self.n_levels):
                if lvl == 0:
                    fused_motion_feature = self.ms_motion_agg[lvl](
                                                    torch.cat([motion_feature_list[lvl], self.block2_inc2x(motion_feature_list[lvl+1]), self.block3_inc4x(motion_feature_list[lvl+2])],dim=1)
                                                    )
                    fused_motion_feature_list.append(fused_motion_feature)       
                elif lvl == 1:
                    fused_motion_feature = self.ms_motion_agg[lvl](
                                                    torch.cat([self.block1_dec2x(motion_feature_list[lvl-1]), motion_feature_list[lvl], self.block3_inc2x(motion_feature_list[lvl+1])],dim=1)
                                                    )
                    fused_motion_feature_list.append(fused_motion_feature)
                elif lvl == 2:
                    fused_motion_feature = self.ms_motion_agg[lvl](
                                                    torch.cat([self.block1_dec4x(motion_feature_list[lvl-2]), self.block2_dec2x(motion_feature_list[lvl-1]), motion_feature_list[lvl]],dim=1)
                                                    )
                    fused_motion_feature_list.append(fused_motion_feature)

        elif self.n_levels == 2:
            for lvl in range(self.n_levels):
                if lvl == 0:
                    fused_motion_feature = self.ms_motion_agg[lvl](
                                                    torch.cat([motion_feature_list[lvl], self.block2_inc2x(motion_feature_list[lvl+1])],dim=1)
                                                    )
                    fused_motion_feature_list.append(fused_motion_feature)       
                elif lvl == 1:
                    fused_motion_feature = self.ms_motion_agg[lvl](
                                                    torch.cat([self.block1_dec2x(motion_feature_list[lvl-1]), motion_feature_list[lvl]],dim=1)
                                                    )
                    fused_motion_feature_list.append(fused_motion_feature)
        else:
            raise Exception()

        assert len(fused_motion_feature_list) == self.n_levels

        return fused_motion_feature_list

class DUCANet_PP(nn.Module):
    def __init__(self, model_cfg, sequence_length):
        super(DUCANet_PP, self).__init__()
        self.model_cfg          = model_cfg
        self.seq_length         = sequence_length
        self.module_category    = self.model_cfg.MODULE_CATEGORY
        self.configs_ducanet    = self.model_cfg.CONFIGS_DUCANET

        d_model = self.configs_ducanet.D_MODEL
        self.configs_ducanet.CONFIGS_ALIGNMENT_BLOCK['D_MODEL']     = d_model
        self.configs_ducanet.CONFIGS_AGGREGATION_BLOCK['D_MODEL']   = d_model
        self.num_attn_layers        = self.configs_ducanet.NUM_ATTN_LAYERS

        self.configs_feature_domain = self.model_cfg.CONFIGS_FEATURE_DOMAIN
        self.target_resolution      = self.configs_feature_domain.TARGET_RESOLUTION
        self.scale_level            = self.configs_feature_domain.SCALE_LEVEL
        self.configs_ducanet.CONFIGS_ALIGNMENT_BLOCK['SCALE_LEVEL'] = self.scale_level
        self.target_channel         = self.configs_feature_domain.TARGET_CHANNEL
        
        self.temporal_src_channel_list = [int(self.target_channel * ratio) for ratio in self.configs_feature_domain.CHANNEL_SCALE_RANGE]
        self.BEV_feature_resolution_list = [int(self.target_resolution * ratio) for ratio in self.configs_feature_domain.RESOLUTION_SCALE_RANGE]    

        BEV_spatial_shape_tuple_list = []
        for _, Size in enumerate(self.BEV_feature_resolution_list):
            BEV_spatial_shape_tuple_list.append([Size, Size])    
        self.BEV_spatial_shape_tuple_list = BEV_spatial_shape_tuple_list
        
        ### Initialization for input source projection layer ####################
        multiscale_src_proj_layer_list = []
        for lvl, input_channel in enumerate(self.temporal_src_channel_list):
            multiscale_src_proj_layer_list.append(nn.Conv2d(input_channel, d_model, kernel_size=1, stride=1,padding=0))
        self.multiscale_src_proj_layers = nn.ModuleList(multiscale_src_proj_layer_list)
        
        ## Initialization for multiple co-attention layers
        coattn_layer        = CoAttn_Layer(cfgs_transformer=self.configs_ducanet, seq_length=self.seq_length)
        self.coattn_layers  = CoAttnLayer_Stack(coattn_layer=coattn_layer, num_layers=self.num_attn_layers, seq_length=self.seq_length, scale_level=self.scale_level)

        ## Initialization for upsampling layers
        self.deblocks = nn.ModuleList()
        upsample_strides = self.configs_feature_domain.UPSAMPLE_STRIDES
        num_upsample_filters = self.configs_feature_domain.NUM_UPSAMPLE_FILTERS
        for lvl in range(self.scale_level):
            stride = upsample_strides[lvl]
            if stride >= 1:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels=d_model, out_channels=num_upsample_filters[lvl],
                                        kernel_size=upsample_strides[lvl], stride=upsample_strides[lvl], bias=False),
                    nn.BatchNorm2d(num_upsample_filters[lvl], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                    )
                )
            else:
                stride = np.round(1/stride).astype(np.int)
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(in_channels=d_model, out_channels=num_upsample_filters[lvl],
                                kernel_size=stride, stride=stride, bias=False),
                    nn.BatchNorm2d(num_upsample_filters[lvl], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                    )
                )

        self.func_unflatten_dict = nn.ModuleDict()
        self.func_unflatten_dict['unflatten_2x'] = nn.Unflatten(dim=-1, unflattened_size=(256, 256))
        self.func_unflatten_dict['unflatten_4x'] = nn.Unflatten(dim=-1, unflattened_size=(128, 128))
        self.func_unflatten_dict['unflatten_8x'] = nn.Unflatten(dim=-1, unflattened_size=(64, 64))

    @staticmethod
    def get_reference_points_DUCA(batch_size, spatial_shape_temporalAttn, n_level_attention, device):
        valid_ratios = torch.ones(batch_size, 1).to(device)
        valid_ratios_temporal_batch = torch.ones(batch_size, 1, n_level_attention, 2).to(device)

        ref_points_deformAttn_list    = []
        for (H_, W_) in spatial_shape_temporalAttn:
            assert H_ == W_
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios * H_)
            ref = torch.stack((ref_x, ref_y), dim=-1)
            ref_points_deformAttn_list.append(ref)
        
        ref_points_deformAttn = torch.cat(ref_points_deformAttn_list, dim=1)
        ref_points_deformAttn = ref_points_deformAttn[:,:,None] * valid_ratios_temporal_batch

        return ref_points_deformAttn

    def forward(self, batch_dict_list):
        target_batch_dict = batch_dict_list[-1]
        cur_device = target_batch_dict['spatial_features'].device
        batch_size = target_batch_dict['spatial_features'].shape[0]

        ## Set up for dual-query
        feature_dict_list = [{} for i in range(self.seq_length)]
        for t, batch_dict in enumerate(batch_dict_list):
            for lvl in range(self.scale_level):
                scale = 2**(lvl+1)
                feature_dict_list[t][f'proj_align_src_{scale}x'] = self.multiscale_src_proj_layers[lvl](batch_dict[f'src_features_for_align_{scale}x'])

        ## Set up for reference points
        ref_points_deformAttn = self.get_reference_points_DUCA(batch_size=batch_size, spatial_shape_temporalAttn=self.BEV_spatial_shape_tuple_list,n_level_attention=self.scale_level, device=cur_device)
        
        ## Set up for spatial shape
        BEV_spatial_shape_tensor = torch.as_tensor(self.BEV_spatial_shape_tuple_list, dtype=torch.long, device=cur_device)
        
        ## Set up for level start index
        temporal_level_start_idx = [0]
        for lvl, Size in enumerate(self.BEV_feature_resolution_list, start=1):
            if lvl != self.scale_level:
                temporal_level_start_idx.append(Size**2)
            else:
                pass
        assert len(temporal_level_start_idx) == self.scale_level
        temporal_level_start_idx = torch.as_tensor(temporal_level_start_idx, dtype=torch.long, device=cur_device).cumsum(dim=0)

        ## Dual-query based co-attention
        encoded_feature = self.coattn_layers(   feature_dict_list   = feature_dict_list,
                                                spatial_shapes      = BEV_spatial_shape_tensor,
                                                level_start_idx     = temporal_level_start_idx,
                                                ref_points          = ref_points_deformAttn     )
        
        ## Upsampling T-QS & channel-wise concatenation
        ups = []
        for lvl in range(self.scale_level):
            scale = 2**(lvl+1)
            ups.append(self.deblocks[lvl](self.func_unflatten_dict[f'unflatten_{scale}x'](torch.transpose(encoded_feature[-1][f'spatialAttn_result_{scale}x'], 1,2))))
        target_batch_dict['aggregated_spatial_features_2d'] = torch.cat(ups, dim=1)

        return target_batch_dict

class DUCANet_CP(nn.Module):  # Temporal-gating transformer module
    def __init__(self, model_cfg, sequence_length):
        super(DUCANet_CP, self).__init__()
        self.model_cfg          = model_cfg
        self.seq_length         = sequence_length
        self.module_category    = self.model_cfg.MODULE_CATEGORY
        self.configs_ducanet    = self.model_cfg.CONFIGS_DUCANET

        d_model = self.configs_ducanet.D_MODEL
        self.configs_ducanet.CONFIGS_ALIGNMENT_BLOCK['D_MODEL']     = d_model
        self.configs_ducanet.CONFIGS_AGGREGATION_BLOCK['D_MODEL']   = d_model
        self.num_attn_layers        = self.configs_ducanet.NUM_ATTN_LAYERS

        self.configs_feature_domain = self.model_cfg.CONFIGS_FEATURE_DOMAIN
        self.target_resolution      = self.configs_feature_domain.TARGET_RESOLUTION
        self.scale_level            = self.configs_feature_domain.SCALE_LEVEL
        self.configs_ducanet.CONFIGS_ALIGNMENT_BLOCK['SCALE_LEVEL'] = self.scale_level
        self.target_channel         = self.configs_feature_domain.TARGET_CHANNEL

        self.temporal_src_channel_list      = [int(self.target_channel * ratio) for ratio in self.configs_feature_domain.CHANNEL_SCALE_RANGE]
        self.BEV_feature_resolution_list   = [int(self.target_resolution * ratio) for ratio in self.configs_feature_domain.RESOLUTION_SCALE_RANGE]    
    
        BEV_spatial_shape_tuple_list = []
        for _, Size in enumerate(self.BEV_feature_resolution_list):
            BEV_spatial_shape_tuple_list.append([Size, Size])    
        self.BEV_spatial_shape_tuple_list = BEV_spatial_shape_tuple_list

        ## Initialization for input source projection layer
        multiscale_src_proj_layer_list = []
        for lvl, input_channel in enumerate(self.temporal_src_channel_list):
            multiscale_src_proj_layer_list.append(nn.Conv2d(input_channel, d_model, kernel_size=1, stride=1,padding=0))
        self.multiscale_src_proj_layers = nn.ModuleList(multiscale_src_proj_layer_list)
        
        ## Initialization for multiple co-attention layers
        coattn_layer        = CoAttn_Layer(cfgs_transformer=self.configs_ducanet, seq_length=self.seq_length)
        self.coattn_layers  = CoAttnLayer_Stack(coattn_layer=coattn_layer, num_layers=self.num_attn_layers, seq_length=self.seq_length, scale_level=self.scale_level)

        ## Initialization for upsampling layers
        self.deblocks = nn.ModuleList()
        upsample_strides = self.configs_feature_domain.UPSAMPLE_STRIDES
        num_upsample_filters = self.configs_feature_domain.NUM_UPSAMPLE_FILTERS
        for lvl in range(self.scale_level):
            stride = upsample_strides[lvl]
            if stride >= 1:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels=d_model, out_channels=num_upsample_filters[lvl],
                            kernel_size=upsample_strides[lvl], stride=upsample_strides[lvl], bias=False),
                    nn.BatchNorm2d(num_upsample_filters[lvl], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                    )
                )
            else:
                stride = np.round(1/stride).astype(np.int)
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(in_channels=d_model, out_channels=num_upsample_filters[lvl],
                            kernel_size=stride, stride=stride, bias=False),
                    nn.BatchNorm2d(num_upsample_filters[lvl], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                    )
                )

        self.func_unflatten_dict = nn.ModuleDict()
        self.func_unflatten_dict['unflatten_2x'] = nn.Unflatten(dim=-1, unflattened_size=(180, 180))
        self.func_unflatten_dict['unflatten_4x'] = nn.Unflatten(dim=-1, unflattened_size=(90, 90))
        
    @staticmethod
    def get_reference_points_DUCA(batch_size, spatial_shape_temporalAttn, n_level_attention, device):
        valid_ratios = torch.ones(2,1).to(device)
        valid_ratios_temporal_batch = torch.ones(batch_size, 1, n_level_attention, 2).to(device)

        ref_points_deformAttn_list    = []
        for (H_, W_) in spatial_shape_temporalAttn:
            assert H_ == W_
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios * H_)
            ref = torch.stack((ref_x, ref_y), dim=-1)
            ref_points_deformAttn_list.append(ref)
        
        ref_points_deformAttn = torch.cat(ref_points_deformAttn_list, dim=1)
        ref_points_deformAttn = ref_points_deformAttn[:,:,None] * valid_ratios_temporal_batch

        return ref_points_deformAttn

    def forward(self, batch_dict_list):
        target_batch_dict = batch_dict_list[-1]
        cur_device = target_batch_dict['spatial_features'].device
        batch_size = target_batch_dict['spatial_features'].shape[0]

        ## Set up for dual-query
        feature_dict_list = [{} for i in range(self.seq_length)]
        for t, batch_dict in enumerate(batch_dict_list):
            for lvl in range(self.scale_level):
                scale = 2**(lvl+1)
                feature_dict_list[t][f'proj_align_src_{scale}x'] = self.multiscale_src_proj_layers[lvl](batch_dict[f'src_features_for_align_{scale}x'])
            
        ## Set up for reference points
        ref_points_deformAttn = self.get_reference_points_DUCA(batch_size=batch_size, spatial_shape_temporalAttn=self.BEV_spatial_shape_tuple_list, n_level_attention=self.scale_level, device=cur_device)
        
        ## Set up for spatial shape
        BEV_spatial_shape_tensor = torch.as_tensor(self.BEV_spatial_shape_tuple_list, dtype=torch.long, device=cur_device)
        
        ## Set up for level start index 
        temporal_level_start_idx = [0]
        for lvl, Size in enumerate(self.BEV_feature_resolution_list, start=1):
            if lvl != self.scale_level:
                temporal_level_start_idx.append(Size**2)
            else:
                pass
        assert len(temporal_level_start_idx) == self.scale_level
        temporal_level_start_idx = torch.as_tensor(temporal_level_start_idx, dtype=torch.long, device=cur_device).cumsum(dim=0)
        
        ## Dual-query based co-attention
        encoded_feature = self.coattn_layers(   feature_dict_list   = feature_dict_list,
                                                spatial_shapes      = BEV_spatial_shape_tensor,
                                                level_start_idx     = temporal_level_start_idx,
                                                ref_points          = ref_points_deformAttn     )
        
        ## Upsampling T-QS & channel-wise concatenation
        ups = []
        for lvl in range(self.scale_level):
            scale = 2**(lvl+1)
            ups.append(self.deblocks[lvl](self.func_unflatten_dict[f'unflatten_{scale}x'](torch.transpose(encoded_feature[-1][f'spatialAttn_result_{scale}x'], 1,2))))
        target_batch_dict['aggregated_spatial_features_2d'] = torch.cat(ups, dim=1)

        return target_batch_dict

class IDANet(nn.Module):
    def __init__(self, cfgs_AttnBlock, num_levels):
        super(IDANet, self).__init__()
        self.n_levels = num_levels
        self.use_motion_context = cfgs_AttnBlock.USE_MOTION_CONTEXT
        
        if self.use_motion_context:
            self.motion_extractor   = FPMNet(in_channels=cfgs_AttnBlock.D_MODEL, num_levels=self.n_levels, scale_list=cfgs_AttnBlock.MOTION_FEATURE_SCALE_LIST)
        else:
            pass
        
        self.deform_attn    = MSDeformAttn(cfgs_AttnBlock.D_MODEL, self.n_levels, cfgs_AttnBlock.NUM_HEAD, cfgs_AttnBlock.NUM_SAMPLING_POINTS)
        self.dropout1       = nn.Dropout(cfgs_AttnBlock.DROPOUT)
        self.norm1          = nn.LayerNorm(cfgs_AttnBlock.D_MODEL)
        self.linear1        = nn.Linear(cfgs_AttnBlock.D_MODEL, cfgs_AttnBlock.DIM_FEEDFORWARD)
        self.activation     = _get_activation_fn(cfgs_AttnBlock.ACTIVATION)
        self.dropout2       = nn.Dropout(cfgs_AttnBlock.DROPOUT)
        self.linear2        = nn.Linear(cfgs_AttnBlock.DIM_FEEDFORWARD, cfgs_AttnBlock.D_MODEL)
        self.dropout3       = nn.Dropout(cfgs_AttnBlock.DROPOUT)
        self.norm2          = nn.LayerNorm(cfgs_AttnBlock.D_MODEL)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src        

    def forward(self, current_feature_dict, past_feature_dict, ref_points, spatial_shapes, level_start_idx):
        if self.use_motion_context:
            motion_feature_list         = self.motion_extractor(current_feature_dict, past_feature_dict)
            deform_query_list_flatten   = [motion_feature.flatten(2).transpose(1,2) for motion_feature in motion_feature_list]
        else:
            deform_query_list_flatten   = [current_feature_dict[f'spatialAttn_result_{2**(idx+1)}x'] for idx in range(self.n_levels)]
            
        deform_query_set = torch.cat(deform_query_list_flatten, dim=1)
        
        key_list = []
        for lvl in range(self.n_levels):
            scale = 2**(lvl+1)
            key_list.append(past_feature_dict[f'temporalAttn_result_{scale}x'])
        assert len(key_list) == self.n_levels

        key_flatten     = torch.cat(key_list, dim=1)
        output          = self.deform_attn(deform_query_set, ref_points, key_flatten, spatial_shapes, level_start_idx)
        attn_result     = self.norm1(key_flatten + self.dropout1(output))

        return self.forward_ffn(attn_result)
        
class IGANet(nn.Module):
    def __init__(self, cfgs_AttnBlock:dict, sequence_length:int, size:int):
        super(IGANet, self).__init__()
        self.in_channel = cfgs_AttnBlock.D_MODEL
        self.seq_length = sequence_length
        self.gating_map_channel = 1
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.conv_gating = nn.ModuleList()
        self.conv_fusion = nn.ModuleList()
        for t in range(self.seq_length-1):
            self.conv_gating.append(nn.Conv2d(self.in_channel*2, self.gating_map_channel, kernel_size=3, padding=1))
            self.conv_fusion.append(nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, padding=1))
        self.agg_all_time = nn.Conv2d(self.in_channel * (self.seq_length-1), self.in_channel, kernel_size=3, padding=1)

        self.func_unflatten = nn.Unflatten(dim=-1, unflattened_size=(size, size))
    
    def forward(self, feature_dict_list, scale):
        target_feature_dict = feature_dict_list[-1]
        temporalAttn_results_2d = [self.func_unflatten(torch.transpose(feature_dict_list[t][f'temporalAttn_result_{scale}x'],1,2)) for t in range(self.seq_length-1)]
        fused_feature_list = []
        
        cur_feature_2d = self.func_unflatten(torch.transpose(target_feature_dict[f'spatialAttn_result_{scale}x'],1,2))
        
        prev_idx = 0
        for gating_layer, fusion_layer in zip(self.conv_gating, self.conv_fusion):
            prev_feature_2d = temporalAttn_results_2d[prev_idx]      
            gating_map_cur = self.sigmoid(gating_layer(torch.cat([prev_feature_2d, cur_feature_2d], 1)))
            gating_map_prev = 1.0 - gating_map_cur
            temp_feature_sum = (gating_map_cur * cur_feature_2d) + (gating_map_prev * prev_feature_2d)
            fused_feature_list.append(self.relu(fusion_layer(temp_feature_sum)))
            prev_idx += 1
        
        if len(fused_feature_list) != (self.seq_length-1):
            raise Exception(f'agg_all_time layer를 들어가기 전 fused_feature_list에는 {self.seq_length-1}개의 tensor가 있어야 합니다.')
        
        temporal_fused_feature = self.relu(self.agg_all_time(torch.cat(fused_feature_list,1)))
        
        if temporal_fused_feature.shape[1] != self.in_channel:
            raise Exception(f'temporal_fused_feature의 Channel수는 {self.target_input_channels}가 되어야 합니다. 현재 {temporal_fused_feature.shape[1]}입니다.')
        
        return temporal_fused_feature.flatten(2).transpose(1,2)

class CoAttn_Layer(nn.Module):
    def __init__(self, cfgs_transformer, seq_length: int):
        super(CoAttn_Layer, self).__init__()
        self.cfgs_alignment_block       = cfgs_transformer.CONFIGS_ALIGNMENT_BLOCK
        self.cfgs_aggregation_block     = cfgs_transformer.CONFIGS_AGGREGATION_BLOCK
        self.seq_length                 = seq_length
        self.n_spatial_levels           = self.cfgs_alignment_block.SCALE_LEVEL

        ## Initialization for inter-frame deformable alignment network (IDANet)        
        align_target_length = self.seq_length-1
        align_module_list = [IDANet(cfgs_AttnBlock=self.cfgs_alignment_block, num_levels=self.n_spatial_levels) for _ in range(align_target_length)]
        self.IDANet_module_list = nn.ModuleList(align_module_list)
        assert len(self.IDANet_module_list) == align_target_length
        
        ## Initialization for inter-frame gated aggregation network (IGANet)
        spatial_size = self.cfgs_aggregation_block.SPATIAL_SIZE
        IGANet_module_list = [IGANet(cfgs_AttnBlock=self.cfgs_aggregation_block, sequence_length=self.seq_length, size=spatial_size[lvl]) for lvl in range(self.n_spatial_levels)]
        self.IGANet_module_list = nn.ModuleList(IGANet_module_list)
        assert len(self.IGANet_module_list) == len(spatial_size)
        
    def forward(self, feature_dict_list, ref_points, spatial_shapes, level_start_idx):
        target_feature_dict = feature_dict_list[-1]

        ## Feature alignment (S-QS Update)
        for t, feature_dict in enumerate(feature_dict_list, start=1):
            if (t == self.seq_length):
                continue
            else:
                output = self.IDANet_module_list[t-1](target_feature_dict, feature_dict, ref_points, spatial_shapes, level_start_idx)

                for lvl in range(self.n_spatial_levels):
                    scale = 2**(lvl+1)
                    if lvl != (self.n_spatial_levels-1):
                        feature_dict[f'temporalAttn_result_{scale}x'] = output[:, level_start_idx[lvl]:level_start_idx[lvl+1], :]
                    else:
                        feature_dict[f'temporalAttn_result_{scale}x'] = output[:, level_start_idx[lvl]:, :]
           
        ## Feature aggregation (T-QS Update)
        for lvl in range(self.n_spatial_levels):
            scale = 2**(lvl+1)
            output = self.IGANet_module_list[lvl](feature_dict_list, scale)
            target_feature_dict[f'spatialAttn_result_{scale}x'] = output
        
        return feature_dict_list

class CoAttnLayer_Stack(nn.Module):
    def __init__(self, coattn_layer, num_layers, seq_length, scale_level):
        super(CoAttnLayer_Stack, self).__init__()
        self.layers     = _get_clones(coattn_layer, num_layers)
        self.seq_length = seq_length
        self.scale_level = scale_level

    def forward(self,feature_dict_list:list, spatial_shapes, level_start_idx, ref_points):
        for t, feature_dict in enumerate(feature_dict_list, start=1):
            for lvl in range(self.scale_level):
                scale = 2**(lvl+1)
                if t == self.seq_length:
                    if feature_dict.get(f'spatialAttn_result_{scale}x') == None: 
                        feature_dict[f'spatialAttn_result_{scale}x'] = feature_dict[f'proj_align_src_{scale}x'].flatten(2).transpose(1,2)
                        feature_dict.pop(f'proj_align_src_{scale}x')
                else:
                    if feature_dict.get(f'temporalAttn_result_{scale}x') == None: 
                        feature_dict[f'temporalAttn_result_{scale}x'] = feature_dict[f'proj_align_src_{scale}x'].flatten(2).transpose(1,2)
                        feature_dict.pop(f'proj_align_src_{scale}x')                    

        for _, layer in enumerate(self.layers):
            feature_dict_list = layer(feature_dict_list, ref_points, spatial_shapes, level_start_idx)

        return feature_dict_list

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
