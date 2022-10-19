from .detector3d_template import Detector3DTemplate

###################################################
from .detector3d_template import Detector3DTemplate_Video

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict


class CenterPoint_Video(Detector3DTemplate_Video):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)
        self.module_list        = self.build_networks()
        self.seq_length         = self.dataset.seq_length
        
        # <self.module_list partition>
        # self.module_common          : Modules classified as 'Common'
        # self.module_target_frame    : Modules classified as 'Target_frame'
        # self.module_multi_frame     : Modules classified as 'MultiFrame_Pipeline'
        
        self.module_common                  = []
        self.module_target_frame            = []
        self.module_multi_frame    = []

        for cur_module in self.module_list:
            category = cur_module.module_category
            if category == 'Common':
                self.module_common.append(cur_module)
            elif category == 'Target_frame':
                self.module_target_frame.append(cur_module)
            elif category == 'MultiFrame_Pipeline':
                self.module_multi_frame.append(cur_module)
            else:
                print(f'{cur_module}의 category : {category}')
                raise Exception('Please check the category where that module object belong to.')
        
    def forward(self, batch_dict_seq_list):
        assert len(batch_dict_seq_list) == self.seq_length
        
        # Forward propagation for common modules
        for batch_dict in batch_dict_seq_list:
            for cur_module in self.module_common:
                batch_dict = cur_module(batch_dict)
        
        # Forward propagation for multi-frame modules
        if len(self.module_multi_frame) != 0:
            for cur_module in self.module_multi_frame:
                batch_dict_target = cur_module(batch_dict_seq_list)
        
        # Forward propagation for target frame modules
        for cur_module in self.module_target_frame:
            batch_dict_target = cur_module(batch_dict_target)
                
        if self.training:
            # Loss function 
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict_target)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict            