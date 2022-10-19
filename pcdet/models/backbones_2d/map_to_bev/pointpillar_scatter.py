import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

class PointPillarScatter_Sequence(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.module_category = self.model_cfg.MODULE_CATEGORY
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros( 
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            # spatial_feature.shape = torch.Size([self.num_bev_features , self.nz * self.nx * self.ny])
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]     # this_coords.shape = torch.Size([해당 frame의 Non empty voxel 갯수 , 4])
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]       # this_coords[:, 1] = 모두 0 (z축), indices : Non-empty voxel이 갖는 Voxelized 공간에서의 인덱스 순번
            indices = indices.type(torch.long)                                                  # indices.shape = 해당 Frame의 Non empty voxel 갯수
            pillars = pillar_features[batch_mask, :]                                            # 해당 Frame에 소속된 Pillar_feature만 가져온다, pillars.shape = torch.Size([해당 frame의 Non empty voxel 갯수 , self.num_bev_features])
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars                                               # indices에 해당하는 Voxel들의 Feature를 기존 Voxelized 공간 상 순번에 Mapping
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # batch_spatial_features.shape = torch.size([Batch, self.num_bev_features, self.nz * self.nx * self.ny])
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        # batch_spatial_features.shape = torch.size([Batch, self.num_bev_features, self.nz , self.ny , self.nx])
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

class PointPillarScatter_InterSweep(nn.Module):
    def __init__(self, model_cfg, grid_size, intersweep_sequence_length, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.module_category = self.model_cfg.MODULE_CATEGORY
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        self.intersweep_seq_length = intersweep_sequence_length

    def forward(self, batch_dict, **kwargs):
        for bin_idx in range(self.intersweep_seq_length):
            pillar_features, coords = batch_dict[f'pillar_features_bin_{bin_idx}'], batch_dict[f'voxel_coords_bin_{bin_idx}']
            batch_spatial_features = []
            batch_size = coords[:, 0].max().int().item() + 1
            for batch_idx in range(batch_size):
                spatial_feature = torch.zeros( 
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)
                # spatial_feature.shape = torch.Size([self.num_bev_features , self.nz * self.nx * self.ny])
                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]     # this_coords.shape = torch.Size([해당 frame의 Non empty voxel 갯수 , 4])
                indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]       # this_coords[:, 1] = 모두 0 (z축), indices : Non-empty voxel이 갖는 Voxelized 공간에서의 인덱스 순번
                indices = indices.type(torch.long)                                                  # indices.shape = 해당 Frame의 Non empty voxel 갯수
                pillars = pillar_features[batch_mask, :]                                            # 해당 Frame에 소속된 Pillar_feature만 가져온다, pillars.shape = torch.Size([해당 frame의 Non empty voxel 갯수 , self.num_bev_features])
                pillars = pillars.t()
                spatial_feature[:, indices] = pillars                                               # indices에 해당하는 Voxel들의 Feature를 기존 Voxelized 공간 상 순번에 Mapping
                batch_spatial_features.append(spatial_feature)

            batch_spatial_features = torch.stack(batch_spatial_features, 0)
            # batch_spatial_features.shape = torch.size([Batch, self.num_bev_features, self.nz * self.nx * self.ny])
            batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
            # batch_spatial_features.shape = torch.size([Batch, self.num_bev_features, self.nz , self.ny , self.nx])
            batch_dict[f'spatial_features_bin_{bin_idx}'] = batch_spatial_features
        return batch_dict        