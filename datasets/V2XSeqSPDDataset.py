import os
import torch
import numpy as np
import torch.utils.data as data
import pickle
from .io import IO
from .build import DATASETS
import logging
from utils import misc


def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        file = pickle.load(f)
        return file

def read_bin_int8(path_file):
    return np.fromfile(path_file, dtype=np.uint8).reshape(-1)


@DATASETS.register_module()
class V2XSeqSPD(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.gt_path = config.GT_PATH
        self.color_path = config.COLOR_PATH
        self.subset = config.subset
        self.pkl_file_path = os.path.join(self.data_root, f'{self.subset}.pkl')
        self.input_num = config.INPUT_POINTS

        print(f'[DATASET] Open file {self.pkl_file_path}')
        file_dict = read_pkl(self.pkl_file_path)
        data_list = file_dict['data_list']

        self.file_list = []
        for item in data_list:
            file_path = item['inf_info']['lidar_path']
            taxonomy_id = item['intersection_loc']
            model_id = item['frame_id']
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': file_path
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def label_remap(self, labels):
        # first map 50 151 class for data preprocess
        labels[labels==0] = 32
        labels[labels==5] = 1
        labels[labels==9] = 17
        labels[labels==13] = 151
        labels[labels==14] = 151
        labels[labels==15] = 151
        labels[labels==16] = 151
        labels[labels==19] = 151
        labels[labels==21] = 151
        labels[labels==27] = 151
        labels[labels==34] = 151
        labels[labels==36] = 136
        labels[labels==38] = 151
        labels[labels==40] = 151
        labels[labels==41] = 151
        labels[labels==42] = 151
        labels[labels==46] = 151
        labels[labels==52] = 151
        labels[labels==53] = 151
        labels[labels==59] = 151
        labels[labels==61] = 151
        labels[labels==66] = 17
        labels[labels==69] = 151
        labels[labels==76] = 151
        labels[labels==82] = 136
        labels[labels==86] = 151
        labels[labels==88] = 151
        labels[labels==90] = 151
        labels[labels==98] = 151
        labels[labels==100] = 151
        labels[labels==104] = 151
        labels[labels==105] = 151
        labels[labels==108] = 151
        labels[labels==115] = 151
        labels[labels==116] = 127
        labels[labels==119] = 151
        labels[labels==125] = 93
        labels[labels==132] = 151
        labels[labels==138] = 93
        labels[labels==139] = 151
        labels[labels==147] = 151
        
        # remap categories to 19 class
        labels[labels==1] = 0
        labels[labels==2] = 1
        labels[labels==4] = 2
        labels[labels==6] = 3
        labels[labels==11] = 4
        labels[labels==12] = 5
        labels[labels==17] = 6
        labels[labels==20] = 7
        labels[labels==32] = 8
        labels[labels==43] = 9
        labels[labels==80] = 10
        labels[labels==83] = 11
        labels[labels==87] = 12
        labels[labels==93] = 13
        labels[labels==102] = 14
        labels[labels==127] = 15
        labels[labels==136] = 16
        labels[labels==150] = 17
        labels[labels==151] = 18
        
        assert np.isin(labels, np.arange(19)).all()
        
        return labels

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        # load points
        partial_data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        sample_idx = misc.random_sample_with_distance_np(partial_data, self.input_num)
        partial_data = partial_data[sample_idx]
        # partial_data = self.pc_norm(partial_data)

        # load gt information
        gt_file_path = os.path.join(self.gt_path, os.path.basename(sample['file_path']))
        gt_label_path = os.path.join(self.color_path, os.path.basename(sample['file_path']))
        gt_data = IO.get(gt_file_path).astype(np.float32)
        gt_label = read_bin_int8(gt_label_path)
        gt_label = self.label_remap(gt_label)

        # return data
        data = {'partial': torch.from_numpy(partial_data).float(), 'gt': torch.from_numpy(gt_data).float(), 'label': torch.from_numpy(gt_label).float()}
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'], data['label'])

    def __len__(self):
        return len(self.file_list)
