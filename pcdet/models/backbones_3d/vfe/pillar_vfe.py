import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ

        self.delete_timestamp = self.model_cfg.DELETE_TIMESTAMP

        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        if self.delete_timestamp:
            num_point_features -= 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
  
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

        # avg_num_points = torch.mean(voxel_num_points).detach().cpu().numpy()
        # file = open("pp_avg_num_points_logging.txt", "a")
        # file.write(str(avg_num_points))
        # file.write('\n')
        # file.close

        # full_voxel_ratio = (torch.sum(voxel_num_points == 20) / len(voxel_num_points)).detach().cpu().numpy()
        # file_1 = open("pp_full_voxel_ratio_logging_retry.txt", "a")
        # file_1.write(str(full_voxel_ratio))
        # file_1.write('\n')
        # file_1.close

        # full_voxel_ratio = (torch.sum(voxel_num_points >= 10) / len(voxel_num_points)).detach().cpu().numpy()
        # file_2 = open("pp_threshold_voxel_ratio_logging.txt", "a")
        # file_2.write(str(full_voxel_ratio))
        # file_2.write('\n')
        # file_2.close

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            if self.delete_timestamp:
                features = [voxel_features[:,:,:-1], f_cluster, f_center]
            else:
                features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict

class PillarVFE_Sequence(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.module_category = self.model_cfg.MODULE_CATEGORY
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
  
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict

class TALayer(nn.Module):
    def __init__(self, dim_ta, reduction_ta):
        super(TALayer, self).__init__()
        self.temporalAttn = nn.Sequential(
                        nn.Linear(in_features=dim_ta, out_features=(dim_ta//reduction_ta)),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_features=(dim_ta//reduction_ta), out_features=dim_ta))

    def forward(self, x):
        out = self.temporalAttn(x)
        return out
        

class CALayer(nn.Module):
    def __init__(self, dim_ca, reduction_ca):
        super(CALayer, self).__init__()
        self.channelAttn = nn.Sequential(
                        nn.Linear(in_features=dim_ca, out_features=(dim_ca//reduction_ca)),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_features=(dim_ca//reduction_ca), out_features=dim_ca))        
    
    def forward(self, x):
        out = self.channelAttn(x)
        return out

class PFE(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True):
        super().__init__()
        
        self.use_norm = use_norm

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        return x

class TCA(nn.Module): # PFN layer with temporal-channel attention
    def __init__(self,
                 tca_configs,
                 in_channels):
        super(TCA, self).__init__()
        self.num_sweeps = tca_configs.NUM_SWEEPS
        self.temporal_reduction_factor, self.channel_reduction_factor = tca_configs.TEMPORAL_REDUCTION_FACTOR, tca_configs.CHANNEL_REDUCTION_FACTOR 
        self.ta = TALayer(dim_ta=self.num_sweeps, reduction_ta=self.temporal_reduction_factor)
        self.ca = CALayer(dim_ca=in_channels, reduction_ca=self.channel_reduction_factor)
        self.sig = nn.Sigmoid()

    def forward(self, input, inputs_temporal_info_dict):
        x = input
        timestep_info = inputs_temporal_info_dict['timestep'].unsqueeze(-1)
        mask_info = inputs_temporal_info_dict['mask'].unsqueeze(-1)
        x_temporal_max_src = torch.max(x, dim=2, keepdim=True)[0]
        x_temporal_max_list = []
        zero_tensor = torch.zeros_like(x_temporal_max_src).to(x_temporal_max_src.device)

        for timestep in range(self.num_sweeps):
            x_max_timestep = torch.where((timestep_info == timestep)&(mask_info == True), x_temporal_max_src, zero_tensor)
            x_max_timestep=torch.max(x_max_timestep, dim=1, keepdim=True)[0]
            x_temporal_max_list.append(x_max_timestep)
        assert len(x_temporal_max_list) == self.num_sweeps
        
        x_channel_max = torch.max(x, dim=1, keepdim=True)[0].squeeze()
        x_temporal_max = torch.cat(x_temporal_max_list, dim=1).squeeze()
        num_voxels, channel = x_channel_max.size()
        ta_weight = self.ta(x_temporal_max).view(num_voxels, self.num_sweeps, 1)
        ca_weight = self.ca(x_channel_max).view(num_voxels, 1, channel)
        cata_weight = torch.mul(ta_weight, ca_weight)
        cata_normal_weight = self.sig(cata_weight) # cata_normal_weight.shape = torch.Size([num_voxels, self.num_sweeps, channel]) ex) N, 10, 64

        timestep_info = timestep_info.squeeze()
        mask_info = mask_info.squeeze()
        output = torch.zeros_like(x).to(x.device)
        for timestep in range(self.num_sweeps):
            output[torch.where((timestep_info==timestep)&(mask_info==True))] = x[torch.where((timestep_info==timestep)&(mask_info==True))] * cata_normal_weight[torch.where((timestep_info==timestep)&(mask_info==True))[0], timestep]

        return output

class PillarVFE_with_stackedTCA_v2(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        
        ######################################################
        if self.model_cfg.WITH_TIMESTEP:
            num_point_features -= 1
        self.tca_configs = self.model_cfg.PFN_WITH_TCA_CONFIGS
        self.add_temporal_period_geometry   = self.model_cfg.ADD_TEMPORAL_PERIOD_GEOMETRY
        self.only_temporal_period_geometry  = self.model_cfg.ONLY_TEMPORAL_PERIOD_GEOMETRY
        self.temporal_encoding_period       = self.model_cfg.TEMPORAL_ENCODING_PERIOD
        ######################################################

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1
        
        ######################################################
        if self.add_temporal_period_geometry:
            if self.only_temporal_period_geometry:
                num_point_feature += 3 * (len(self.model_cfg.TEMPORAL_ENCODING_PERIOD) - 1)
            else:
                for i in range(len(self.model_cfg.TEMPORAL_ENCODING_PERIOD)):
                    num_point_features += 3
        ######################################################
        
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        
        if self.model_cfg.USE_PFN_WITH_TCA:
            self.num_tca_stack = self.tca_configs.NUM_STACK
            num_filters = [num_point_features] + list(self.num_filters*self.num_tca_stack)
            assert len(num_filters) == int(1 + self.num_tca_stack)
        else:
            num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        
        if self.model_cfg.USE_PFN_WITH_TCA:
            tca_layers = []
            for i in range(self.num_tca_stack):
                in_filters = num_filters[i]
                out_filters = num_filters[i + 1]
                pfn_layers.append(
                    PFE(in_channels=in_filters, out_channels=out_filters, use_norm=self.use_norm)
                )
                tca_layers.append(
                    TCA(tca_configs=self.tca_configs, in_channels=num_filters[i+1])
                )
            self.pfn_layers = nn.ModuleList(pfn_layers)
            self.tca_layers = nn.ModuleList(tca_layers)

            if self.tca_configs.USE_FC:
                self.FC1 = nn.Sequential(nn.Linear(num_filters[-1]*2, num_filters[-1]),
                                        nn.ReLU(inplace=True),
                                    )
            if self.tca_configs.USE_CHANNEL_COMPRESSION:
                out_channels = self.tca_configs.TARGET_CHANNEL
                self.FC_Compression =   nn.Sequential(nn.Linear(num_filters[-1], out_channels),
                                        nn.ReLU(inplace=True)
                                    )
        else:
            pfn_layers.append(PFE(in_channels=num_filters[0], out_channels=num_filters[1], use_norm=self.use_norm))
            self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        if self.tca_configs.USE_CHANNEL_COMPRESSION:
            return self.tca_configs.TARGET_CHANNEL
        else:
            return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        # voxel_features.shape = torch.Size([num of total voxels, torch.max(voxel_num_points), 5]) / 5 : (x,y,z,r,t)
        # coords.shape=  torch.Size([Sum of voxels in minibatch, 4]) / 4 : (index_in_minibatch, Z-axis coord, Y-axis coord, X-axis coord) / Ex) Batch size: 3 ->> index_in_minibatch: 0,1,2
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        
        if self.use_absolute_xyz:
            if self.add_temporal_period_geometry:
                f_cluster_per_temporal_period = []
                for encoding_period in self.model_cfg.TEMPORAL_ENCODING_PERIOD:
                    f_cluster_per_temporal_period.append(batch_dict[f'f_cluster_per_{encoding_period}sweeps'])
                
                if self.only_temporal_period_geometry:
                    features = [voxel_features[:,:,:5]] + f_cluster_per_temporal_period
                    features.append(f_center)
                else:
                    features = [voxel_features[:,:,:5], f_cluster] + f_cluster_per_temporal_period
                    features.append(f_center)
            else:
                features = [voxel_features[:,:,:5], f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1) # (self.with_distance -> Fales) , features.shape = torch.Size([num of total voxels, torch.max(voxel_num_points), 11])

        if self.model_cfg.USE_PFN_WITH_TCA:
            voxel_count = features.shape[1]
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
            # mask.shape = torch.Size([num of total voxels, torch.max(voxel_num_points)]), mask[target_voxel_idx, empty_point_idx] = False
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
            features *= mask

            inputs_temporal_info_dict ={}
            inputs_temporal_info_dict['mask'] = mask.squeeze()
            inputs_temporal_info_dict['timestep'] = voxel_features[:,:,-1]
            
            for layer_idx in range(self.num_tca_stack):
                if layer_idx == 0:
                    pfe_result = self.pfn_layers[layer_idx](features)
                    tca_result = self.tca_layers[layer_idx](pfe_result, inputs_temporal_info_dict)
                    if self.tca_configs.USE_RESIDUAL_CONNECTION:
                        out = pfe_result + tca_result
                    elif self.tca_configs.USE_FC:
                        out = torch.cat([pfe_result, tca_result],dim=2)
                        out = self.FC1(out)
                    else:
                        out = tca_result
                    
                else:
                    pfe_result = self.pfn_layers[layer_idx](out)
                    tca_result = self.tca_layers[layer_idx](pfe_result, inputs_temporal_info_dict)
                    out = tca_result
            
            if self.tca_configs.USE_CHANNEL_COMPRESSION:
                out = self.FC_Compression(out)
        else:
            out = self.pfn_layers[0](features)

        features_max = torch.max(out, dim=1, keepdim=True)[0]
        features_max = features_max.squeeze()
        
        if self.tca_configs.USE_CHANNEL_COMPRESSION:
            batch_dict['voxel_features'] = features_max
        else:
            batch_dict['pillar_features'] = features_max
        return batch_dict

class PillarVFE_with_stackedTCA_Sequence_v2(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        
        self.module_category = self.model_cfg.MODULE_CATEGORY
        ######################################################
        if self.model_cfg.WITH_TIMESTEP:
            num_point_features -= 1
        self.tca_configs = self.model_cfg.PFN_WITH_TCA_CONFIGS
        self.add_temporal_period_geometry   = self.model_cfg.ADD_TEMPORAL_PERIOD_GEOMETRY
        self.only_temporal_period_geometry  = self.model_cfg.ONLY_TEMPORAL_PERIOD_GEOMETRY
        self.temporal_encoding_period       = self.model_cfg.TEMPORAL_ENCODING_PERIOD
        ######################################################

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1
        
        ######################################################
        if self.add_temporal_period_geometry:
            if self.only_temporal_period_geometry:
                num_point_feature += 3 * (len(self.model_cfg.TEMPORAL_ENCODING_PERIOD) - 1)
            else:
                for i in range(len(self.model_cfg.TEMPORAL_ENCODING_PERIOD)):
                    num_point_features += 3
        ######################################################
        
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        
        if self.model_cfg.USE_PFN_WITH_TCA:
            self.num_tca_stack = self.tca_configs.NUM_STACK
            num_filters = [num_point_features] + list(self.num_filters*self.num_tca_stack)
            assert len(num_filters) == int(1 + self.num_tca_stack)
        else:
            num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        
        if self.model_cfg.USE_PFN_WITH_TCA:
            tca_layers = []
            for i in range(self.num_tca_stack):
                in_filters = num_filters[i]
                out_filters = num_filters[i + 1]
                pfn_layers.append(
                    PFE(in_channels=in_filters, out_channels=out_filters, use_norm=self.use_norm)
                )
                tca_layers.append(
                    TCA(tca_configs=self.tca_configs, in_channels=num_filters[i+1])
                )
            self.pfn_layers = nn.ModuleList(pfn_layers)
            self.tca_layers = nn.ModuleList(tca_layers)

            if self.tca_configs.USE_FC:
                self.FC1 = nn.Sequential(nn.Linear(num_filters[-1]*2, num_filters[-1]),
                                        nn.ReLU(inplace=True),
                                    )
            if self.tca_configs.USE_CHANNEL_COMPRESSION:
                out_channels = self.tca_configs.TARGET_CHANNEL
                self.FC_Compression =   nn.Sequential(nn.Linear(num_filters[-1], out_channels),
                                        nn.ReLU(inplace=True)
                                    )
        else:
            pfn_layers.append(PFE(in_channels=num_filters[0], out_channels=num_filters[1], use_norm=self.use_norm))
            self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        if self.tca_configs.USE_CHANNEL_COMPRESSION:
            return self.tca_configs.TARGET_CHANNEL
        else:
            return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        # voxel_features.shape = torch.Size([num of total voxels, torch.max(voxel_num_points), 5]) / 5 : (x,y,z,r,t)
        # coords.shape=  torch.Size([Sum of voxels in minibatch, 4]) / 4 : (index_in_minibatch, Z-axis coord, Y-axis coord, X-axis coord) / Ex) Batch size: 3 ->> index_in_minibatch: 0,1,2
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        
        if self.use_absolute_xyz:
            if self.add_temporal_period_geometry:
                f_cluster_per_temporal_period = []
                for encoding_period in self.model_cfg.TEMPORAL_ENCODING_PERIOD:
                    f_cluster_per_temporal_period.append(batch_dict[f'f_cluster_per_{encoding_period}sweeps'])
                
                if self.only_temporal_period_geometry:
                    features = [voxel_features[:,:,:5]] + f_cluster_per_temporal_period
                    features.append(f_center)
                else:
                    features = [voxel_features[:,:,:5], f_cluster] + f_cluster_per_temporal_period
                    features.append(f_center)
            else:
                features = [voxel_features[:,:,:5], f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1) # (self.with_distance -> Fales) , features.shape = torch.Size([num of total voxels, torch.max(voxel_num_points), 11])

        if self.model_cfg.USE_PFN_WITH_TCA:
            voxel_count = features.shape[1]
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
            # mask.shape = torch.Size([num of total voxels, torch.max(voxel_num_points)]), mask[target_voxel_idx, empty_point_idx] = False
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
            features *= mask

            inputs_temporal_info_dict ={}
            inputs_temporal_info_dict['mask'] = mask.squeeze()
            inputs_temporal_info_dict['timestep'] = voxel_features[:,:,-1]
            
            for layer_idx in range(self.num_tca_stack):
                if layer_idx == 0:
                    pfe_result = self.pfn_layers[layer_idx](features)
                    tca_result = self.tca_layers[layer_idx](pfe_result, inputs_temporal_info_dict)
                    if self.tca_configs.USE_RESIDUAL_CONNECTION:
                        out = pfe_result + tca_result
                    elif self.tca_configs.USE_FC:
                        out = torch.cat([pfe_result, tca_result],dim=2)
                        out = self.FC1(out)
                    else:
                        out = tca_result
                    
                else:
                    pfe_result = self.pfn_layers[layer_idx](out)
                    tca_result = self.tca_layers[layer_idx](pfe_result, inputs_temporal_info_dict)
                    out = tca_result
            
            if self.tca_configs.USE_CHANNEL_COMPRESSION:
                out = self.FC_Compression(out)
        else:
            out = self.pfn_layers[0](features)

        features_max = torch.max(out, dim=1, keepdim=True)[0]
        features_max = features_max.squeeze()
        
        if self.tca_configs.USE_CHANNEL_COMPRESSION:
            batch_dict['voxel_features'] = features_max
        else:
            batch_dict['pillar_features'] = features_max
        return batch_dict

class PillarVFE_InterSweep(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, intersweep_sequence_length, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.module_category = self.model_cfg.MODULE_CATEGORY
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.intersweep_seq_length = intersweep_sequence_length

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        # voxel_features.shape = torch.Size([num of total voxels, torch.max(voxel_num_points), 5]) / 5 : (x,y,z,r,t)
        # coords.shape=  torch.Size([Sum of voxels in minibatch, 4]) / 4 : (index_in_minibatch, Z-axis coord, Y-axis coord, X-axis coord) / Ex) Batch size: 3 ->> index_in_minibatch: 0,1,2
        for bin_idx in range(self.intersweep_seq_length):
            voxel_features, voxel_num_points, coords = batch_dict[f'voxels_bin_{bin_idx}'], batch_dict[f'voxel_num_points_bin_{bin_idx}'], batch_dict[f'voxel_coords_bin_{bin_idx}']
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            f_cluster = voxel_features[:, :, :3] - points_mean

            f_center = torch.zeros_like(voxel_features[:, :, :3])
            f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

            if self.with_distance:
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            features = torch.cat(features, dim=-1) # (self.with_distance -> Fales) , features.shape = torch.Size([num of total voxels, torch.max(voxel_num_points), 11])

            voxel_count = features.shape[1]
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
            # mask.shape = torch.Size([num of total voxels, torch.max(voxel_num_points)]), mask[target_voxel_idx, empty_point_idx] = False
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
            features *= mask
            for pfn in self.pfn_layers:
                features = pfn(features)
            features = features.squeeze()
            batch_dict[f'pillar_features_bin_{bin_idx}'] = features

        return batch_dict