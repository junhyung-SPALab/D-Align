from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
############################################################
from .anchor_head_multi import AnchorHeadMulti_Sequence
from .center_head import CenterHead_Sequence
from .aux_loss_head import Heatmap_Similarity_Loss_Head
from .anchor_head_multi import AnchorHeadMulti_InterSweep

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,

    ##################################
    'AnchorHeadMulti_Sequence': AnchorHeadMulti_Sequence,
    'CenterHead_Sequence': CenterHead_Sequence,
    'Heatmap_Similarity_Loss_Head': Heatmap_Similarity_Loss_Head,
    'AnchorHeadMulti_InterSweep': AnchorHeadMulti_InterSweep,
}
