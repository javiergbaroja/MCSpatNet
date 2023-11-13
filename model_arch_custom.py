import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import collections
from distutils.util import strtobool;
import numpy as np  
import sys
from utils import create_logger
from sa_net_arch_utilities_pytorch import CNNArchUtilsPyTorch;

from model_arch import UnetVggMultihead
import math

logger = create_logger('UnetVggMultihead_custom')

def dice_loss(pred:torch.Tensor, target:torch.Tensor, smooth:float=1e-3, reduction:str='mean')->torch.Tensor:
    """Assuming of shape BxCxHxW."""
    if reduction =='sum':
        intse = torch.sum(pred * target, (0, 2, 3)) # sum of intersection per class
        union = torch.sum(pred**2, (0, 2, 3)) + torch.sum(target**2, (0, 2, 3)) # sum of union per class
        loss = 1.0 - (2.0 * intse + smooth) / (union + smooth) # loss per class
        loss = loss.sum() # loss (aggregation over classes)
        
    else:
        intse = torch.sum(pred * target, (2,3)) # sum of intersection per mask and class
        union = torch.sum(pred**2, (2, 3)) + torch.sum(target**2, (2, 3)) # sum of union per mask and class
        loss = 1.0 - (2.0 * intse + smooth) / (union + smooth) # loss per mask and class
        loss = loss.sum(1) # loss per mask (aggregation over classes)

    # if math.isnan(pred.sum()):
    #     print('Nan pred', file=sys.stderr)
    # if math.isnan(target.sum()):
    #     print('Nan target', file=sys.stderr)
    # if math.isnan(intse.sum()):
    #     print('Nan intersect', file=sys.stderr)
    # if math.isnan(union.sum()):
    #     print('Nan union', file=sys.stderr)
    # if math.isnan(loss.sum()):
    #     print('Nan loss', file=sys.stderr)
    
    return loss

class UnetVggMultihead_custom(UnetVggMultihead):
    def __init__(self, load_weights=False, kwargs=None):
        super().__init__(load_weights, kwargs)

    def forward_net(self, x, feat_indx_list, feat_as_dict):
        '''
            x: input image normalized by dividing by 255
            feat_indx_list: list of indices corresponding to features generated at different model blocks.
                If list is not empty, then the features listed will be returned
                feature_code = {'decoder':0, 'cell-detect':1, 'class':2, 'subclass':3, 'k-cell':4}
            feat_as_dict: if feat_indx_list is not empty, the features indicated in the list will be returned in the form of a dictonary, where key is features index identifier and value is the features
        '''
        feat = None
        feat_dict = {}
        feat_indx = 0
        encoder_out = [];
        for l in self.encoder:     
            x = l(x);
            encoder_out.append(x);
        x = self.bottleneck(x);
        j = len(self.decoder);
        for l in self.decoder:            
            x = l[0](x);
            j -= 1;
            corresponding_layer_indx = j;

            ## crop and concatenate
            if(j > 0):
                cropped = CNNArchUtilsPyTorch.crop_a_to_b(encoder_out[corresponding_layer_indx],  x);
                x = torch.cat((cropped, x), 1) ;
            for i in range(1, len(l)):
                x = l[i](x);


        # Check if decoder features will be returned in output
        if(feat_indx in feat_indx_list):
            if(feat_as_dict):
                feat_dict[feat_indx] = x#.detach().cpu().numpy()
            else:
                feat = x#.detach().cpu().numpy()
        
        c=[]
        # f=None
        for layer in self.final_layers_lst:
            feat_indx += 1
            f1 = layer[0](x) # output features from current head
            c.append(layer[1](f1)) # output prediction from current head
            # if(f is None):
            #     f = f1
            # else:
            #     f = torch.cat((f1, f), 1) ;

            # Check if current head features will be returned in output
            if(feat_indx in feat_indx_list):
                if(feat_as_dict):
                    feat_dict[feat_indx] = f1#.detach().cpu().numpy()
                else:
                    if(feat is None):
                        feat = f1#.detach().cpu().numpy()
                    else:
                        feat= torch.cat((feat, f1),dim=1)#np.concatenate((feat, f1.detach().cpu().numpy()), axis=1)
        return c, feat_dict if feat_as_dict else feat


    def forward(self, x:torch.Tensor, gt:dict=dict(), loss_fns:dict=dict(), feat_indx_list:list=[], feat_as_dict:bool=False, phase:str='train'):
        """passes input batch through the network and computes the loss

        Args:
            x (torch.Tensor): batch of input images
            gt (torch.Tensor): list of the ground truth maps
                'gt_dmap_all':torch.Tensor
                'gt_dmap':torch.Tensor
                'gt_dmap_subclasses':torch.Tensor
                'gt_kmap': dict
                    'tensor':torch.Tensor
                    'r_classes_all':int
            loss_fns (list): loss functions
                'criterion_l1_sum'
                'criterion_sig'
                'criterion_softmax'
            feat_indx_list (list, optional): list of indices corresponding to features generated at different model blocks.
            feat_as_dict (bool, optional): if feat_indx_list is not empty, the features indicated in the list will be returned in the form of a dictonary, where key is features index identifier and value is the features. Defaults to False.
            phase (str, optional): 'train' or 'test'. Defaults to 'train'.
        """

        et_dmap_lst, feats = self.forward_net(x, feat_indx_list, feat_as_dict)
        # If no features requested, then just return predictions list

        if len(feat_indx_list) != 0:
            return et_dmap_lst, feats

        et_dmap_all=et_dmap_lst[0][:,:,2:-2,2:-2] # The cell detection prediction
        et_dmap_class=et_dmap_lst[1][:,:,2:-2,2:-2] # The cell classification prediction
        et_dmap_subclasses= et_dmap_lst[2][:,:,2:-2,2:-2] # The cell clustering sub-class prediction
        et_kmap=et_dmap_lst[3][:,:,2:-2,2:-2]**2   # The cross K-functions estimation

        del et_dmap_lst

        # Apply K function loss only on the detection mask regions
        k_loss_mask = gt['gt_dmap_all'].clone() 
        loss_l1_k = loss_fns['criterion_l1_sum'](et_kmap*(k_loss_mask), gt['gt_kmap']['tensor']*(k_loss_mask)) 

        # Apply Sigmoid and Softmax activations to the detection and classification predictions, respectively.
        et_all_sig = loss_fns['criterion_sig'](et_dmap_all)
        et_class_sig = loss_fns['criterion_softmax'](et_dmap_class)

        # Compute Dice loss on the detection and classification predictions
        loss_dice_class = dice_loss(et_class_sig, gt['gt_dmap'], reduction='mean')
        loss_dice_all = dice_loss(et_all_sig, gt['gt_dmap_all'], reduction='mean')

        if phase == 'test':
            return loss_dice_all, loss_dice_class, loss_l1_k, et_all_sig, et_class_sig
        
        et_subclasses_sig = loss_fns['criterion_softmax'](et_dmap_subclasses)
        loss_dice_subclass = dice_loss(et_subclasses_sig, gt['gt_dmap_subclasses'], reduction='mean')

        return loss_dice_all, loss_dice_class, loss_dice_subclass, loss_l1_k
    
