# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from .ms_deform_attn import MSDeformAttn, MSDeformAttn_wo_value_proj, MSDeformAttn_STATrans, MSDeformAttn_STATrans_one_shot, MSDeformAttn_STATrans_one_shot_v2, MSDeformAttn_STATrans_Pairwise, MSDeformAttn_STATrans_Pairwise_v2
from .ms_deform_attn import MSDeformAttn_STATrans_v2
