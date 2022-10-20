import pickle
from tqdm import tqdm

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

data_path = f'./data/nuscenes/v1.0-trainval/'
pickle_file_name = 'nuscenes_dbinfos_10sweeps_withvelo_add_GT_box_token.pkl'
with open(data_path + pickle_file_name,'rb') as f:
    dbinfos_data = pickle.load(f)
my_dict = {}

for class_name in class_name_list:
    my_dict[class_name] = {}

for class_name in class_name_list:
    for db_sample in tqdm(dbinfos_data[class_name]):
        key = db_sample['gt_box_token']
        my_dict[class_name][key] = db_sample
    print(f'Finish the class {class_name}')
assert len(my_dict) == len(dbinfos_data)

with open(f'{data_path}sample_box_token_ref_dbinfos.pkl', 'wb') as file:
    pickle.dump(my_dict, file, protocol=pickle.HIGHEST_PROTOCOL)