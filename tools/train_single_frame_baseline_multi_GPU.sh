# Train a PointPillars model using 2 GPUs
# ex) CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --batch_size 4

CUDA_VISIBLE_DEVICES={List up multiple GPUs index} python3 -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file {Your config file path} --batch_size {Mini batch_size}