from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torchvision.transforms as T
import random

from utils import parse_json_file, create_logger, get_npy_files
logger = create_logger('CellsDataset-KFunc')



class CellsDataset(Dataset):
    def __init__(self,
                 img_root:str, 
                 class_indx:str, 
                 gt_dmap_subclasses_root:str, 
                 split_filepath:str=None, 
                 phase:str='train', 
                 fixed_size:int=-1, 
                 max_side:int=-1, 
                 max_scale:int=-1):
        """Dataset. It returns the image, the ground truth dilated dot map, the ground truth dot map, the pseudo-ground truth dilated dot map used for the deep clustering module, 
        the pseudo-ground truth dot map used for the deep clustering module, the ground truth cross k-function map, and the image name.

        Args:
            img_root (str): path to the image folder
            class_indx (str): a comma separated list of channel indices to return from the ground truth
            gt_dmap_subclasses_root (str): path to the folder containing the pseudo-ground truth dilated dot maps used for the deep clustering module.
            gt_dots_subclasses_root (str): path to the folder containing the pseudo-ground truth dot maps used for the deep clustering module.
            gt_kmap_root (str): path to the folder containing the ground truth cross k-function maps.
            split_filepath (str, optional): path to JSON file containing the information about the validation partition. If none, all the samples will be used. Defaults to None.
            phase (str, optional): 'train' or 'test'. Defaults to 'train'.
            fixed_size (int, optional): If >0, the dataset will pass crops of a fixed size during training. Defaults to -1.
            max_side (int, optional): indicates whether to have a maximum side length during training. Defaults to -1.
            max_scale (int, optional): apply padding to make the patch side divisible by max_scale. Defaults to -1.
        """

        super(CellsDataset, self).__init__()
        
        self.gt_dmap_subclasses_root=gt_dmap_subclasses_root
        self.phase=phase

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

    def _apply_hor_flip(self, arr:np.ndarray):
        return arr[:,::-1].copy()
    def _apply_ver_flip(self, arr:np.ndarray):
        return arr[::-1,:].copy()
    
    def __len__(self):
        return self.n_samples
    
    def _augment(self, img:np.ndarray, gt_dmap:np.ndarray, gt_subclasses_dmap:np.ndarray, gt_kmap:np.ndarray):
        if random.random() > 0.5:
            vertical = T.RandomVerticalFlip(1)
            img = vertical(img)
            gt_dmap = vertical(gt_dmap)
            gt_subclasses_dmap = vertical(gt_subclasses_dmap)
            gt_kmap = vertical(gt_kmap)
        if random.random() > 0.5:
            horizontal = T.RandomHorizontalFlip(1)
            img = horizontal(img)
            gt_dmap = horizontal(gt_dmap)
            gt_subclasses_dmap = horizontal(gt_subclasses_dmap)
            gt_kmap = horizontal(gt_kmap)

        if self.max_side > 0:
            h = img.shape[1]
            w = img.shape[2]
            h2 = h
            w2 = w
            crop = False
            if(h > self.max_side):
                h2 = self.max_side
                crop = True
            if(w > self.max_side):
                w2 = self.max_side
                crop = True
            if(crop):
                y=0
                x=0
                if(not (h2 ==h)):
                    y = np.random.randint(0, high = h-h2)
                if(not (w2 ==w)):
                    x = np.random.randint(0, high = w-w2)
                img = img[y:y+h2, x:x+w2, :]
                gt_dmap = gt_dmap[y:y+h2, x:x+w2]
                gt_subclasses_dmap = gt_subclasses_dmap[y:y+h2, x:x+w2]
                gt_kmap = gt_kmap[y:y+h2, x:x+w2]

        if self.fixed_size < 0:
            crop = T.RandomCrop((img.shape[1]//4, img.shape[2]//4)) 
        else:
            crop = T.RandomCrop((min(self.fixed_size,img.shape[1]), min(self.fixed_size,img.shape[2]))) 
        
        img = crop(img)
        gt_dmap = crop(gt_dmap)
        gt_subclasses_dmap = crop(gt_subclasses_dmap)
        gt_kmap = crop(gt_kmap)
        return img, gt_dmap, gt_subclasses_dmap, gt_kmap


        
    def _apply_padding(self, img, gt_dmap, gt_dots, gt_subclasses_dmap, gt_kmap):
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
        gt_subclasses_dmap = pad_zero(gt_subclasses_dmap) if self.phase == 'train' else None
        gt_kmap = pad_zero(gt_kmap)

        return img, gt_dmap, gt_dots, gt_subclasses_dmap, gt_kmap


    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        # Read image, normalize it, and make sure it is in RGB format
        img_name=self.img_names[index]
        array = torch.from_numpy(np.load(img_name, allow_pickle=True).astype(np.float32))

        img = array[...,:3] / 255.
        gt_dots = array[...,5:8]
        gt_dmap = array[...,8:11]
        gt_kmap = array[...,11:32] 
        
        # Read pseudo ground truth class sub clusters dilated dot maps
        if self.phase == 'train':
            name_aux = os.path.sep.join(os.path.normpath(img_name).split(os.path.sep)[-2:])
            gt_subclasses_path = os.path.join(self.gt_dmap_subclasses_root,name_aux.replace('.png','.npy'));
            gt_subclasses_path2 = os.path.join(self.gt_dmap_subclasses_root,name_aux.replace('.png','.png.npy'));
            # logger.info(gt_subclasses_path)
            if(os.path.isfile(gt_subclasses_path)):
                gt_subclasses_dmap=np.load(gt_subclasses_path, allow_pickle=True).squeeze()
            elif(os.path.isfile(gt_subclasses_path2)):
                gt_subclasses_dmap=np.load(gt_subclasses_path2, allow_pickle=True).squeeze()
            else:
                gt_subclasses_dmap=np.zeros((img.shape[0], img.shape[1], len(self.class_indx_list)*5))
        else:
            gt_subclasses_dmap = None
        gt_subclasses_dmap = torch.from_numpy(gt_subclasses_dmap.astype(np.float32)) if self.phase == 'train' else None


        # Covert image and ground truth to pytorch format
        img=img.permute((2,0,1)) # convert to order (channel,rows,cols)
        gt_kmap=gt_kmap.permute((2,0,1)) # convert to order (channel,rows,cols)
        if(len(self.class_indx_list) > 1):
            gt_dmap=gt_dmap.permute((2,0,1)) # convert to order (channel,rows,cols)
            gt_dots=gt_dots.permute((2,0,1)) if self.phase == 'test' else None# convert to order (channel,rows,cols)
            gt_subclasses_dmap=gt_subclasses_dmap.permute((2,0,1)) if self.phase == 'train' else None # convert to order (channel,rows,cols)
        else: 
            gt_dmap=gt_dmap[...,:,:]
            gt_dots=gt_dots[...,:,:]  if self.phase == 'test' else None
            gt_subclasses_dmap=gt_subclasses_dmap[...,:,:] if self.phase == 'train' else None

        if self.phase == 'train':
            img, gt_dmap, gt_subclasses_dmap, gt_kmap = self._augment(img, gt_dmap, gt_subclasses_dmap, gt_kmap)

        # Add padding to make sure image dimensions are divisible by max_scale
        if self.max_scale>1: # to downsample image and density-map to match deep-model.
            img, gt_dmap, gt_dots, gt_subclasses_dmap, gt_kmap = self._apply_padding(img, gt_dmap, gt_dots, gt_subclasses_dmap, gt_kmap)
        
        # if gt_kmap.shape[0] != 21:
        #     print(f'Error at {img_name}')

        if self.phase == 'train':
            return img,gt_dmap, gt_subclasses_dmap, gt_kmap, [img_name]
        else:
            return img,gt_dmap,gt_dots, gt_kmap, [img_name]

