import argparse
import json
import pickle
from tqdm import tqdm

idx = None
temp_token_list = []

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version for making sequence pickle file')
    parser.add_argument('--target_split', type=str, default='train', help='Target split to be applied(train or val)')
    parser.add_argument('--seq_length', type=int, default=None, help='The length of sequence')
    args = parser.parse_args()
    return args

def collect_keyframe_token(keyframe_idx = None, seq_length = 0):
    for i in range(len(json_data)):
        if json_data[i]['token'] == pickle_data[keyframe_idx]['token']:
            cur_scene_token = json_data[i]['scene_token']
            idx = i
            break

    temp_token_list.clear()
    for n in range(seq_length):
        if json_data[idx-n]['prev'] != "" and json_data[idx-n]['scene_token'] == cur_scene_token:
            temp_token_list.insert(0, json_data[idx-n]['token'])
        elif json_data[idx-n]['prev'] == "" and json_data[idx-n]['scene_token'] == cur_scene_token:
            if len(temp_token_list) != 0:
                for k in range(seq_length-n):
                    temp_token_list.insert(0, json_data[idx-n]['token'])
                break
            else:
                for k in range(seq_length):
                    temp_token_list.insert(0, json_data[idx-n]['token'])
                break

    if len(temp_token_list) != seq_length:
        print(len(temp_token_list))
        raise Exception('Not enough seq_length has been saved in that list.')
    
    return temp_token_list

def collect_pickle_data(token_list = None):
    seq_list = []
    for token in token_list:
        for i in range(len(pickle_data)):
            if pickle_data[i]['token'] == token:
                seq_list.append(pickle_data[i])
                break
    
    for info in seq_list:
        if not isinstance(info, dict):
            raise Exception("The elements in seq_list are not dictionary.")

    return seq_list

if __name__ == '__main__':
    args = parse_config()
    seq_data_pickle = []
    version         = args.version
    target_split    = args.target_split
    seq_length      = args.seq_length

    data_path           = f'./data/nuscenes/{version}/'
    pickle_file_name    = f'nuscenes_infos_10sweeps_{target_split}.pkl'
    json_file_name      = 'sample.json'

    with open(data_path + pickle_file_name, 'rb') as f_pickle:
        pickle_data = pickle.load(f_pickle)
    with open(f'{data_path}{version}/{json_file_name}', 'r') as f_json:
        json_data = json.load(f_json)
    print('Fininsh to load original pickle data and token json file\n')
    
    for i, _ in enumerate(tqdm(pickle_data)):
        token_list  = collect_keyframe_token(keyframe_idx=i, seq_length=seq_length)
        seq_list    = collect_pickle_data(token_list=token_list)
        
        if len(seq_list) != seq_length:
            raise Exception("")
        else:
            seq_data_pickle.append(seq_list)
    assert len(seq_data_pickle) == len(pickle_data)

    ## Dumping with new pickle file ##
    print("Dumping for pickle file\n")
    if pickle_file_name == 'nuscenes_infos_10sweeps_train.pkl':
        print('train sample: %d' % len(seq_data_pickle))
        with open(f'{data_path}nuscenes_seq_{str(seq_length)}_infos_10sweeps_train.pkl', 'wb') as f:
            pickle.dump(seq_data_pickle, f)
        
    elif pickle_file_name == 'nuscenes_infos_10sweeps_val.pkl':
        print('val sample: %d' % len(seq_data_pickle))
        with open(f'{data_path}nuscenes_seq_{str(seq_length)}_infos_10sweeps_val.pkl', 'wb') as f:
            pickle.dump(seq_data_pickle, f)
    elif pickle_file_name == 'nuscenes_infos_10sweeps_test.pkl':
        print('val sample: %d' % len(seq_data_pickle))
        with open(f'{data_path}nuscenes_seq_{str(seq_length)}_infos_10sweeps_test.pkl', 'wb') as f:
            pickle.dump(seq_data_pickle, f)
    else:
        raise Exception('There is no such case.')
    