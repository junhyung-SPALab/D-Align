### Prepare NuScenes Dataset
* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and 
organize the downloaded files as follows: 
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval
├── pcdet
├── tools
```
* Install the `nuscenes-devkit` with version `1.0.5` by running the following command: 
```shell script
pip install nuscenes-devkit==1.0.5
```
### Create data infos for single frame pipeline
* Generate the data infos by running the following command (it may take several hours)
* An argument 'include_video_pipeline' for applying the ground truth sampling augmentation to the multi-frame pipeline has been added to the original command in OpenPCDet.
```
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval --include_video_pipeline True
```

### Create data infos for multi frame pipeline
*  Create info pickle files required for training and evaluation of the multi frame pipeline.
    *  seq_length: N (target frame and N-1 previous frames)
    *  The following shows the command line for creating a pickle file for a sequence length of 3 frames for the training and validation
    ```
    python create_sequence_pickle.py --version v1.0-trainval --target_split train --seq_length 3
    python create_sequence_pickle.py --version v1.0-trainval --target_split val --seq_length 3
    ```
*  Create ground truth database info pickle files required for applying the ground truth sampling augmentation in training stage.
    *  Create ground truth database sample token dictionary
    ```
    python create_gt_box_token_dict.py
    ```
    *  Create dbinfos for (t-1)th frame
    ```
    python create_sequential_dbinfos_for_train.py --ref_dbinfos nuscenes_dbinfos_10sweeps_withvelo_add_GT_box_token --target_idx 1
    ```
    *  Create dbinfos for (t-2)th frame
    ```
    python create_sequential_dbinfos_for_train.py --ref_dbinfos temporal_idx_t-1_dbinfos_D-Align --target_idx 2
    ```    
