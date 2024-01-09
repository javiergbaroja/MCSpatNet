import torch
import numpy as np  
from utils import create_logger
from sa_net_arch_utilities_pytorch import CNNArchUtilsPyTorch;

from model_arch import UnetVggMultihead
from loss_functions import asym_unified_focal_loss, ce_plus_dice_loss

logger = create_logger('UnetVggMultihead_custom')

class UnetVggMultihead_custom(UnetVggMultihead):
    def __init__(self, load_weights=False, kwargs=None):
        super().__init__(load_weights, kwargs)
        self.use_dice_loss = kwargs['use_dice_loss']
        self.use_ce_loss = kwargs['use_ce_loss']
        self.use_paper_loss = True
        if not self.use_dice_loss and not self.use_ce_loss:
            raise ValueError('At least one of dice or cross-entropy loss must be used for cell detection and classification branches')
        if self.use_paper_loss:
            self.loss_func_seg = asym_unified_focal_loss()
        else:
            self.loss_func_seg = ce_plus_dice_loss(reduction='mean', dice=self.use_dice_loss, ce=self.use_ce_loss)

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
                'gt_kmap': torch.Tensor
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

        if len(feat_indx_list) != 0 or phase=='infer':
            return et_dmap_lst, feats

        et_dmap_all=et_dmap_lst[0][:,:,2:-2,2:-2] # The cell detection prediction
        et_dmap_class=et_dmap_lst[1][:,:,2:-2,2:-2] # The cell classification prediction
        et_dmap_subclasses= et_dmap_lst[2][:,:,2:-2,2:-2] # The cell clustering sub-class prediction
        et_kmap=et_dmap_lst[3][:,:,2:-2,2:-2]**2   # The cross K-functions estimation

        del et_dmap_lst

        # Apply K function loss only on the detection mask regions
        k_loss_mask = gt['gt_dmap_all'].clone() 
        loss_l1_k = loss_fns['criterion_l1_sum'](et_kmap*(k_loss_mask), gt['gt_kmap']*(k_loss_mask)) 

        # Apply Sigmoid and Softmax activations to the detection and classification predictions, respectively.
        et_all_sig = loss_fns['criterion_sig'](et_dmap_all)
        et_class_sig = loss_fns['criterion_softmax'](et_dmap_class)

        # Compute Dice loss on the detection and classification predictions
        loss_dice_class = self.loss_func_seg(et_class_sig, gt['gt_dmap'])
        loss_dice_all = self.loss_func_seg(et_all_sig, gt['gt_dmap_all'])

        if phase == 'test':
            return loss_dice_all, loss_dice_class, loss_l1_k, et_all_sig, et_class_sig
        
        et_subclasses_sig = loss_fns['criterion_softmax'](et_dmap_subclasses)
        loss_dice_subclass = self.loss_func_seg(et_subclasses_sig, gt['gt_dmap_subclasses'])

        return loss_dice_all, loss_dice_class, loss_dice_subclass, loss_l1_k
    
