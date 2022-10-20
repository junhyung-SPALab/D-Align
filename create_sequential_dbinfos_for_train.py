import pickle
import json
import copy
import argparse
import pdb
from tqdm import tqdm

from nuscenes import NuScenes
nusc = NuScenes(version="v1.0-trainval", dataroot="./data/nuscenes/v1.0-trainval", verbose=True)

class_name_list=[
'traffic_cone',
'truck',
'car',
'pedestrian',
'ignore',
'construction_vehicle',
'barrier',
'motorcycle',
'bicycle',
'bus',
'trailer'
]

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--ref_dbinfos', type=str, default=None, help='')
    parser.add_argument('--target_idx', type=int, default=None, help='')
    
    args = parser.parse_args()    
    return args

def collect_temporal_dbinfos(sample_box_token_dbinfos:dict, new_dbinfo_data:dict, ref_dbinfo_data:dict):
    for class_name in class_name_list:
        new_dbinfo_data[class_name] = []
        print(f'Start to prev idx process for {class_name}')
        for db_sample in tqdm(ref_dbinfo_data[class_name]):
            cur_gt_token = db_sample['gt_box_token']
            cur_sample_dict = nusc.get('sample_annotation', cur_gt_token)
            temp_gt_token = cur_sample_dict['prev']

            if temp_gt_token != '':
                prev_sample = sample_box_token_dbinfos[class_name].get(temp_gt_token, None)
                if prev_sample != None:
                    new_dbinfo_data[class_name].append(copy.deepcopy(prev_sample))
                else:
                    new_dbinfo_data[class_name].append(copy.deepcopy(db_sample))
            elif temp_gt_token == '':
                new_dbinfo_data[class_name].append(copy.deepcopy(db_sample))
            else:
                NotImplementedError()
        print(f'class {class_name} finished')                    
    
    return new_dbinfo_data

if __name__ == '__main__':
    args = parse_config()

    seq_data_pickle = []
    ref_dbinfos_file_name   = args.ref_dbinfos
    target_idx              = args.target_idx

    sample_box_ref_path     = f'./sample_box_token_ref_dbinfos.pkl'
    pickle_path             = f'./{ref_dbinfos_file_name}.pkl'

    data_path           = f'./data/nuscenes/v1.0-trainval/'    
    with open(data_path + sample_box_ref_path, 'rb') as f_pickle:
        sample_box_ref_data = pickle.load(f_pickle)
    
    with open(data_path + pickle_path, 'rb') as f_pickle:
        pickle_data = pickle.load(f_pickle)

    print('Fininsh to load pickle files\n')
    
    new_dbinfo_dict = {}
    new_dbinfos_update = collect_temporal_dbinfos(  sample_box_token_dbinfos=sample_box_ref_data,
                                                    new_dbinfo_data=new_dbinfo_dict,
                                                    ref_dbinfo_data=pickle_data
                                                )

    with open(f'{data_path}temporal_idx_t-{target_idx}_dbinfos_D-Align.pkl', 'wb') as f:
        pickle.dump(new_dbinfos_update, f)
    
    print('Fininsh to create temporal dbinfos pickle file\n')