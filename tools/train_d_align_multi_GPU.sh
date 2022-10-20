# Train a D-Align_PP using 2 GPUs
# ex) CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file cfgs/D_Align_models/D_Align_PP.yaml --batch_size 4 --pretrained_model {File path for pretrained weight}

CUDA_VISIBLE_DEVICES={List up multiple GPUs index} python3 -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file {Your config file path} --batch_size {Mini batch_size} --pretrained_model {File path for pretrained weight}