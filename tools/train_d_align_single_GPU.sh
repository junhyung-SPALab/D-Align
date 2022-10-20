# Train a D-Align_PP using single GPU
# python3 train.py --cfg_file cfgs/D_Align_models/D_Align_PP.yaml --batch_size 4 --pretrained_model {File path for pretrained weight} 

python3 train.py --cfg_file {Your config file path} --batch_size {Mini batch_size} --pretrained_model {File path for pretrained weight}