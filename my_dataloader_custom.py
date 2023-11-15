from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import crop
import random

from utils import parse_json_file, create_logger, get_npy_files
logger = create_logger('CellsDataset')

class CellsDataset(Dataset):
    def __init__(self, img_root, class_indx, split_filepath=None, phase='train', fixed_size=-1, max_side=-1, max_scale=-1, return_padding=False):
        super(CellsDataset, self).__init__()
        '''
        img_root: the root path of img.
        class_indx: a comma separated list of channel indices to return from the ground truth
        split_filepath: if not None, then use only the images in the file
        phase: train or test
        fixed_size:  if > 0 return crops of side=fixed size during training
        max_side: boolean indicates whether to have a maximum side length during training
        max_scale: apply padding to make the patch side divisible by max_scale
        return_padding: return the x and y padding added by max_scale
        '''

        self.phase=phase
        self.return_padding = return_padding
        self.root_dir = img_root

        self.img_names = []
        for file in parse_json_file(split_filepath):
            self.img_names.extend(get_npy_files(os.path.join(img_root, file['patches_dir'])))
            
        self.n_samples=len(self.img_names)

        self.fixed_size = fixed_size
        self.max_side = max_side
        self.max_scale = max_scale
        self.class_indx = class_indx
        self.class_indx_list = [int(x) for x in self.class_indx.split(',')]
        self.id = 0

    def get_worker_id(self, worker_id, worker_seed):
        random.seed(worker_seed)
        self.id = worker_id

    def _augment(self, img:torch.Tensor, gt_dmap:torch.Tensor, gt_dots:torch.Tensor):
        if random.random() > 0.5:
            vertical = T.RandomVerticalFlip(1)
            img = vertical(img)
            gt_dmap = vertical(gt_dmap)
            gt_dots = vertical(gt_dots)
        if random.random() > 0.5:
            horizontal = T.RandomHorizontalFlip(1)
            img = horizontal(img)
            gt_dmap = horizontal(gt_dmap)
            gt_dots = horizontal(gt_dots)

        if self.max_side > 0:
            h = img.shape[1]
            w = img.shape[2]
            h2 = h
            w2 = w
            do_crop = False
            if(h > self.max_side):
                h2 = self.max_side
                do_crop = True
            if(w > self.max_side):
                w2 = self.max_side
                do_crop = True
            if do_crop:
                y=0
                x=0
                if(not (h2 ==h)):
                    y = np.random.randint(0, high = h-h2)
                if(not (w2 ==w)):
                    x = np.random.randint(0, high = w-w2)
                img = img[y:y+h2, x:x+w2, :]
                gt_dmap = gt_dmap[y:y+h2, x:x+w2]
                gt_dots = gt_dots[y:y+h2, x:x+w2]

        if self.fixed_size < 0:
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(img.shape[1]//4, img.shape[2]//4)) 
        else:
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(min(self.fixed_size,img.shape[1]), min(self.fixed_size,img.shape[2])))  
        
        img = crop(img, i, j, h, w)
        gt_dmap = crop(gt_dmap, i, j, h, w)
        gt_dots = crop(gt_dots, i, j, h, w)

        return img, gt_dmap, gt_dots


        
    def _apply_padding(self, img:torch.Tensor, gt_dmap:torch.Tensor, gt_dots:torch.Tensor):
        ds_rows=int(img.shape[1]//self.max_scale)*self.max_scale
        ds_cols=int(img.shape[2]//self.max_scale)*self.max_scale
        pad_y1 = 0
        pad_y2 = 0
        pad_x1 = 0
        pad_x2 = 0
        if(ds_rows < img.shape[1]):
            pad_y1 = (self.max_scale - (img.shape[1] - ds_rows))//2
            pad_y2 = (self.max_scale - (img.shape[1] - ds_rows)) - pad_y1
        if(ds_cols < img.shape[2]):
            pad_x1 = (self.max_scale - (img.shape[2] - ds_cols))//2
            pad_x2 = (self.max_scale - (img.shape[2] - ds_cols)) - pad_x1

        pad_one = T.Pad((pad_x1, pad_y1, pad_x2, pad_y2), padding_mode='constant', fill=1)
        pad_zero = T.Pad((pad_x1, pad_y1, pad_x2, pad_y2), padding_mode='constant', fill=0)

        img = pad_one(img)
        gt_dmap = pad_zero(gt_dmap)
        gt_dots = pad_zero(gt_dots) if self.phase == 'test' else None

        return img, gt_dmap, gt_dots, (pad_y1, pad_y2, pad_x1, pad_x2)
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'

        # Read image, normalize it, and make sure it is in RGB format
        img_name=self.img_names[index]
        array = torch.from_numpy(np.load(img_name, allow_pickle=True).astype(np.float32))
        img = array[...,:3] / 255.
        gt_dots = array[...,5:8]
        gt_dmap = array[...,8:11]

        img=img.permute((2,0,1)) # convert to order (channel,rows,cols)
        if(len(self.class_indx_list) > 1):
            gt_dmap=gt_dmap.permute((2,0,1)) # convert to order (channel,rows,cols)
            gt_dots=gt_dots.permute((2,0,1)) # convert to order (channel,rows,cols)
        else:
            gt_dmap=gt_dmap[np.newaxis,:,:]
            gt_dots=gt_dots[np.newaxis,:,:]

        if self.phase == 'train':
            img, gt_dmap, gt_dots = self._augment(img, gt_dmap, gt_dots)
        
        if self.max_scale>1: # to downsample image and density-map to match deep-model.
            img, gt_dmap, gt_dots, padding_coords = self._apply_padding(img, gt_dmap, gt_dots)
        
        # Covert image and ground truth to pytorch format
        
        if self.return_padding:
            return img, gt_dmap, gt_dots, img_name, padding_coords
        else:
            return img, gt_dmap, gt_dots, img_name


