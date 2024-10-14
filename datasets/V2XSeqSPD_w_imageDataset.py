import os
import torch
import numpy as np
import torch.utils.data as data
import pickle
from .io import IO
from .build import DATASETS
import logging
from PIL import Image
from utils import misc
from mmengine.fileio import get
import mmcv
from typing import List, Optional, Tuple, Union, no_type_check

def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        file = pickle.load(f)
        return file

def read_bin_int8(path_file):
    return np.fromfile(path_file, dtype=np.uint8).reshape(-1)





@DATASETS.register_module()
class V2XSeqSPDWithImage(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.gt_path = config.GT_PATH
        self.color_path = config.COLOR_PATH
        self.subset = config.subset
        self.pkl_file_path = os.path.join(self.data_root, f'{self.subset}.pkl')
        self.input_num = config.INPUT_POINTS
        self.img_resize = tuple(config.img_resize_shape)

        print(f'[DATASET] Open file {self.pkl_file_path}')
        file_dict = read_pkl(self.pkl_file_path)
        data_list = file_dict['data_list']

        self.file_list = []
        for item in data_list:
            file_path = item['inf_info']['lidar_path']
            image_path = item['inf_info']['image_path']
            taxonomy_id = item['intersection_loc']
            model_id = item['frame_id']
            first_key = list(item['images'].keys())[0]
            lidar2cam = item['images'][first_key]['lidar2cam']
            cam2img = item['images'][first_key]['cam2img']
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': file_path,
                'image_path': image_path,
                'lidar2cam': lidar2cam,
                'cam2img': cam2img
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
        # first map to 151 class
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
        
        # then map to 19 class
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

    def get_image(self, filename):
        img_bytes = get(filename, backend_args=None)
        img = mmcv.imfrombytes(
            img_bytes, flag='color', backend='cv2', channel_order='rgb')
        return img

    def _scale_size(
        self, 
        size: Tuple[int, int],
        scale: Union[float, int, tuple],
    ) -> Tuple[int, int]:
        """Rescale a size by a ratio.

        Args:
            size (tuple[int]): (w, h).
            scale (float | tuple(float)): Scaling factor.

        Returns:
            tuple[int]: scaled size.
        """
        if isinstance(scale, (float, int)):
            scale = (scale, scale)
        w, h = size
        return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


    def resize_img(self, img, cam2img_calib):
        img, w_scale, h_scale = mmcv.imresize(
            img,
            self.img_resize,
            interpolation="bilinear",
            return_scale=True,
            backend=None)
        
        return img, (w_scale, h_scale)

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        partial_data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        sample_idx = misc.random_sample_with_distance_np(partial_data, self.input_num)
        partial_data = partial_data[sample_idx]
        # partial_data = self.pc_norm(partial_data)
        
        gt_file_path = os.path.join(self.gt_path, os.path.basename(sample['file_path']))
        gt_label_path = os.path.join(self.color_path, os.path.basename(sample['file_path']))
        image_path = os.path.join(self.pc_path, sample['image_path'])
        gt_data = IO.get(gt_file_path).astype(np.float32)
        gt_label = read_bin_int8(gt_label_path)
        gt_label = self.label_remap(gt_label)
        # image = Image.open(image_path)
        # image_np = np.array(image)
        image_np = self.get_image(image_path)
        image_np, scale_factor = self.resize_img(image_np, sample['cam2img'])
        # Update camera intrinsics
        sample['cam2img'][0] *= np.array(scale_factor[0])
        sample['cam2img'][1] *= np.array(scale_factor[1])
        
        # gt_data = self.pc_norm(gt_data)

        # return data
        data = {'partial': torch.from_numpy(partial_data).float(),
                'gt': torch.from_numpy(gt_data).float(),
                'label': torch.from_numpy(gt_label).float(),
                'image': torch.from_numpy(image_np).permute(2, 0, 1).float()}
        return sample['taxonomy_id'], sample['model_id'], sample['lidar2cam'], sample['cam2img'], (data['partial'], data['gt'], data['label'], data['image'])

    def __len__(self):
        return len(self.file_list)
