#!/usr/bin/env python3.9

import shutil
import sys
import os
import glob
import math
from tqdm import tqdm
from natsort import natsorted
from termcolor import colored
from collections import Counter
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import cv2
from skimage import filters
from skimage.measure import label, moments, regionprops_table

from cluster_helper import *
from utils import get_train_args, create_logger, empty_trash, seed_everything
from model_arch_custom import UnetVggMultihead_custom as UnetVggMultihead

from my_dataloader_w_kfunc_custom import CellsDataset_train, CellsDataset_test
from my_dataloader_custom import CellsDataset as CellsDataset_simple
from cluster_helper_eff import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hover_net.metrics.stats_utils import pair_coordinates

feature_code = {'decoder':0, 'cell-detect':1, 'class':2, 'subclass':3, 'k-cell':4}

def worker_init_fn(worker_id):
    # ! to make the seed chain reproducible, must use the torch random, not numpy
    # the torch rng from main thread will regenerate a base seed, which is then
    # copied into the dataloader each time it created (i.e start of each epoch)
    # then dataloader with this seed will spawn worker, now we reseed the worker
    worker_info = torch.utils.data.get_worker_info()
    # to make it more random, simply switch torch.randint to np.randint
    worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
    # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
    # retrieve the dataset copied into this worker process
    # then set the random seed for each augmentation
    worker_info.dataset.get_worker_id(worker_id, worker_seed)
    return


if __name__=="__main__":
    args = get_train_args()
    logger = create_logger()
    logger.info(f'------Training arguments print-out:\n{args}')

    # checkpoints_save_path: path to save checkpoints
    checkpoints_save_path   = os.path.join(args.checkpoints_root_dir, args.checkpoints_folder_name)
    cluster_tmp_out         = os.path.join(args.root_dir, args.clustering_pseudo_gt_root, args.checkpoints_folder_name)
    
    if args.model_param_path is None:
        if os.path.isdir(checkpoints_save_path):
            shutil.rmtree(args.checkpoints_root_dir)
        if os.path.isdir(cluster_tmp_out):
            shutil.rmtree(cluster_tmp_out)
    else:
        chekpoint = os.path.join(checkpoints_save_path, 'mcspat_last.pth')
        args.model_param_path = chekpoint if os.path.isfile(chekpoint) else None

    os.makedirs(args.checkpoints_root_dir, exist_ok=True)
    os.makedirs(checkpoints_save_path, exist_ok=True)
    os.makedirs(args.clustering_pseudo_gt_root, exist_ok=True)
    os.makedirs(cluster_tmp_out, exist_ok=True)
    
    start_epoch             = 0  # To use if continuing training from a previous epoch loaded from model_param_path
    epoch_start_eval_prec   = 0 # After epoch_start_eval_prec epochs start to evaluate F-score of predictions on the validation set.
    restart_epochs_freq     = 50 # reset frequency for optimizer
    next_restart_epoch      = restart_epochs_freq + start_epoch
    gpu_or_cpu              ='cuda' if torch.cuda.is_available() else 'cpu' # use cuda or cpu
    nr_gpu = torch.cuda.device_count() if gpu_or_cpu == 'cuda' else 1
    device=torch.device(gpu_or_cpu)
    seed_everything(args.seed) # set seed for reproducibility
    
    # Configure training dataset
    train_dmap_subclasses_root = cluster_tmp_out

    # Configure validation dataset
    test_dmap_subclasses_root = cluster_tmp_out

    dropout_prob = 0.2
    initial_pad = 126 # We add padding so that final output has same size as input since we do not use same padding conv.
    interpolate = 'False'
    conv_init = 'he'

    n_channels = 3
    n_classes = 3 # number of cell classes (others, healthy, malignant)
    n_classes_out = n_classes + 1 # number of output classes = number of cell classes (lymphocytes, tumor, stromal) + 1 (for cell detection channel)
    class_indx = '1,2,3' # the index of the classes channels in the ground truth
    n_clusters = 5 # number of clusters per class
    n_classes_deep_clustering = n_clusters * n_classes # number of output classes for the cell cluster classification

    prints_per_epoch=1 # print frequency per epoch

    # Initialize the range of the radii for the K function for each class
    r_step = 15
    r_range = range(0, 100, r_step)
    r_arr = np.array([*r_range])
    r_classes = len(r_range) # number of output channels for the K function for a single class
    k_func_dim = r_classes * n_classes # number of output channels for the K function over all classes

    k_norm_factor = 100 # the maximum K-value (i.e. number of nearby cells at radius r) to normalize the K-func to [0,1]
    lamda_class = args.lamda_class;  # weight for dice loss for cell classification
    lamda_detect = args.lamda_detect # weight for dice loss for cell detection 
    lamda_subclasses = args.lamda_subclass # weight for dice loss for secondary output channels (cell cluster classification)
    lamda_k = args.lamda_kfunc # weight for L1 loss for K function regression

    model=nn.DataParallel(UnetVggMultihead(kwargs={'dropout_prob':dropout_prob, 
                                                   'initial_pad':initial_pad, 
                                                   'interpolate':interpolate, 
                                                   'conv_init':conv_init, 
                                                   'n_classes':n_classes, 
                                                   'n_channels':n_channels, 
                                                   'n_heads':4, 
                                                   'head_classes':[1, n_classes, n_classes_deep_clustering, k_func_dim],
                                                   'use_dice_loss':args.use_dice_loss,
                                                   'use_ce_loss':args.use_ce_loss}))
    if args.model_param_path is not None:
        if args.model_param_path.split('_')[-2].isnumeric():
            start_epoch = int(args.model_param_path.split('_')[-2])
            model.load_state_dict(torch.load(args.model_param_path, map_location=device), strict=False);
        else:
            net_dict = torch.load(args.model_param_path, map_location=device)
            start_epoch = net_dict['epoch']
            model.load_state_dict(net_dict['desc'], strict=False);
            del net_dict
        centroids = np.load(os.path.join(checkpoints_save_path, 'centroids_last.npy'), allow_pickle=True)
    else:
        centroids = None
    
    model.to(device)
    model.module.control_encoder_and_bottleneck(order='freeze', print_info=True)
    logger.info(f'------Loaded model from {args.model_param_path} [epoch {start_epoch}]. Encoder and bottleneck layers are frozen.')

    # Initialize sigmoid layer for cell detection
    criterion_sig = nn.Sigmoid()
    # Initialize softmax layer for cell classification
    criterion_softmax = nn.Softmax(dim=1)
    # Initialize L1 loss for K function regression
    criterion_l1_sum = nn.L1Loss(reduction='sum')

    # Initialize Optimizer
    optimizer=torch.optim.Adam(model.parameters(), args.lr[0])

    # Initialize training dataset loader
    train_dataset=CellsDataset_train(args.root_dir,class_indx, train_dmap_subclasses_root, split_filepath=args.train_split_filepath, phase='train', fixed_size=args.crop_size, max_scale=16)
    train_loader=DataLoader(train_dataset, batch_size=args.batch_size[0], shuffle=True, num_workers=14, worker_init_fn=worker_init_fn)

    # Validation set does not crop images (fixed_size=-1) --> they are of different sizes --> batch size must be 1
    test_dataset=CellsDataset_test(args.root_dir, class_indx, split_filepath=args.test_split_filepath, phase='test', fixed_size=-1, max_scale=16)
    test_loader=DataLoader(test_dataset, batch_size=int(args.batch_size[0]*4/3), shuffle=False, num_workers=14, worker_init_fn=worker_init_fn)

    # Initialize training dataset loader for clustering phase
    simple_train_dataset=CellsDataset_simple(args.root_dir, class_indx, split_filepath=args.cluster_data_filepath, phase='test', fixed_size=-1, max_scale=16, return_padding=True)
    simple_train_loader=DataLoader(simple_train_dataset, batch_size=int(args.batch_size[0]*4/3), shuffle=False, num_workers=14, worker_init_fn=worker_init_fn)

    logger.info(f'------Training set size: {len(train_dataset)}. Dataloader has {len(train_loader)} batches of size {train_loader.batch_size}. Learning rate is {args.lr[0]}.')
    logger.info(f'------Clustering set size: {len(simple_train_dataset)}. Dataloader has {len(simple_train_loader)} batches of size {simple_train_loader.batch_size}.')
    logger.info(f'------Validation set size: {len(test_dataset)}. Dataloader has {len(test_loader)} batches of size {test_loader.batch_size}.')
    logger.info(f'------Number of cell classes: {n_classes}')
    logger.info(f'------Checkpoint save path: {checkpoints_save_path}')
    logger.info(f'------Cluster pseudo-labels save path: {cluster_tmp_out}')
    logger.info(f'------Number of GPUs: {nr_gpu}')
    # Use prints_per_epoch to get iteration number to generate sample output
    print_frequency_test = len(test_loader) // prints_per_epoch;

    best_epoch_filepath=None
    best_epoch=None
    best_f1_mean = 0
    best_prec_recall_diff = math.inf
    # scaler = torch.cuda.amp.GradScaler()
    
    # Define LR scheduling
    lr_update_epoch = [int(args.start_finetuning*3/4), int(args.epochs*3/4)]
    lr_bool_updates = [False, False]

    with tqdm(range(start_epoch,args.epochs), unit='epoch', desc="Training", dynamic_ncols=True) as epoch_iterator:
        for epoch in epoch_iterator:
            # If epoch already exists then skip
            epoch_files = glob.glob(os.path.join(checkpoints_save_path, 'mcspat_epoch_'+str(epoch)+"_*.pth"))
            if len(epoch_files) > 0:
                continue;
            # Cluster features at the beginning of each epoch
            logger.info(f'EPOCH {epoch}')
            logger.info('------Commencing clustering for pseudo-label generation...')
            centroids = perform_clustering(model, simple_train_loader, n_clusters, n_classes, [feature_code['k-cell'], feature_code['subclass']], train_dmap_subclasses_root, centroids, using_crops=args.cropped_data)
            logger.info('------Clustering for pseudo-label generation complete.')
                    
            # Training phase definitions
            if epoch + 1 >= args.start_finetuning and not model.module.is_finetuning:
                model.module.control_encoder_and_bottleneck(order='finetune')
                train_loader=DataLoader(train_dataset, batch_size=args.batch_size[1], shuffle=True, num_workers=14, worker_init_fn=worker_init_fn)
                logger.info(f"------{colored('Commencing finetuning of encoder', color='red', attrs=['bold'])}")
                logger.info(f'------Training set size: {len(train_dataset)}. Dataloader has {len(train_loader)} batches of size {train_loader.batch_size}. Learning rate is {args.lr[1]}.')
                for g in optimizer.param_groups:
                    g['lr'] = args.lr[1]

            elif epoch + 1 >= lr_update_epoch[0] and not lr_bool_updates[0]:
                for g in optimizer.param_groups:
                    g['lr'] /= 2
                lr_bool_updates[0] = True
                logger.info(f"------Learning rate updated to {g['lr']}.")
                
            elif epoch + 1 >= lr_update_epoch[1] and not lr_bool_updates[1]:
                for g in optimizer.param_groups:
                    g['lr'] /= 2
                lr_bool_updates[1] = True
                logger.info(f"------Learning rate updated to {g['lr']}.")


            model.train()

            # Initialize variables for accumulating loss over the epoch
            epoch_loss=0
            epoch_loss_detect=0
            epoch_loss_class=0
            epoch_loss_kfunc=0
            epoch_loss_subclass=0
            train_count_k = 0

            with tqdm(train_loader, unit='batch', desc=f"Train Step [{epoch}]", ascii=' =', dynamic_ncols=True) as batch_loader:
                for (img, gt_dmap, gt_dmap_subclasses, gt_kmap, __) in batch_loader:
                    ''' 
                        img: input image
                        gt_dmap: ground truth map for cell classes (lymphocytes, epithelial/tumor, stromal) with dilated dots. This can be a binary mask or a density map ( in which case it will be converted to a binary mask)
                        gt_dots: ground truth binary dot map for cell classes (lymphocytes, epithelial/tumor, stromal). 
                        gt_dmap_subclasses: ground truth map for cell clustering sub-classes with dilated dots. This can be a binary mask or a density map ( in which case it will be converted to a binary mask) 
                        gt_dots_subclasses: ground truth binary dot map for cell clustering sub-classes. 
                        gt_kmap: ground truth k-function map. At each cell center contains the cross k-functions centered at that cell. 
                        img_name: img filename
                    '''
                    gt_kmap /= k_norm_factor # Normalize K functions ground truth
                    
                    # Convert ground truth maps to binary mask (in case they were density maps)
                    gt_dmap = gt_dmap > 0
                    gt_dmap_subclasses = gt_dmap_subclasses > 0
                    # Get the detection ground truth maps from the classes ground truth maps
                    gt_dmap_all =  gt_dmap.max(dim=1, keepdim=True)[0]
                    # Set datatype and move to GPU
                    gt_dmap = gt_dmap.type(torch.FloatTensor)
                    gt_dmap_all = gt_dmap_all.type(torch.FloatTensor)
                    gt_dmap_subclasses = gt_dmap_subclasses.type(torch.FloatTensor)
                    gt_kmap = gt_kmap.type(torch.FloatTensor)

                    # forward propagation
                    loss_dice_all, loss_dice_class, loss_dice_subclass, loss_l1_k = model(x=img,
                                                                                        gt={'gt_dmap_all':gt_dmap_all,
                                                                                            'gt_dmap':gt_dmap,
                                                                                            'gt_dmap_subclasses':gt_dmap_subclasses,
                                                                                            'gt_kmap':gt_kmap},
                                                                                        loss_fns={'criterion_l1_sum':criterion_l1_sum,
                                                                                                  'criterion_sig':criterion_sig,
                                                                                                  'criterion_softmax':criterion_softmax})
                    
                    ################### MEAN REDUCTION ###################
                    loss_dice_all = loss_dice_all.mean()
                    loss_dice_class = loss_dice_class.mean()
                    loss_dice_subclass = loss_dice_subclass.mean()
                    loss_l1_k = loss_l1_k.sum() / (gt_dmap_all.clone().sum()*k_func_dim) # fake mean reduction over pixels that should be non-zero (focus region)
                    ###########################################################################

                    loss_dice = lamda_class*loss_dice_class + lamda_detect*loss_dice_all + lamda_subclasses*loss_dice_subclass

                    # Add up the dice loss and the K function L1 loss. The K function can be NAN especially in the beginning of training. Do not add to loss if it is NAN.
                    loss = loss_dice 
                    if(not math.isnan(loss_l1_k.item())):
                        loss += loss_l1_k * lamda_k
                        train_count_k += 1
                        epoch_loss_kfunc += loss_l1_k.item()

                    # Backpropagate loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # scaler.scale(loss).backward()
                    # scaler.step(optimizer)
                    # scaler.update()

                    epoch_loss += loss.item()
                    epoch_loss_detect += loss_dice_all.item()
                    epoch_loss_class += loss_dice_class.item()
                    epoch_loss_subclass += loss_dice_subclass.item()

                    # CHANGE THIS FOR TQDM UPDATES
                    batch_loader.set_postfix({"DiceLoss": loss_dice.item(), "K-FuncLoss":loss_l1_k.item()})
            del img, gt_dmap, gt_dmap_subclasses, gt_kmap
            empty_trash()
            epoch_loss = epoch_loss/len(train_loader)
            epoch_loss_detect = epoch_loss_detect/len(train_loader)
            epoch_loss_class = epoch_loss_class/len(train_loader)
            epoch_loss_subclass = epoch_loss_subclass/len(train_loader)
            epoch_loss_kfunc = epoch_loss_kfunc/train_count_k
            logger.info(f'------Loss print-out (epoch mean):')
            logger.info(f'------------train-loss-detect: {epoch_loss_detect:.4f}')
            logger.info(f'------------train-loss-class: {epoch_loss_class:.4f}')
            logger.info(f'------------train-loss-subclass: {epoch_loss_subclass:.4f}')
            logger.info(f'------------train-loss-kfunc: {epoch_loss_kfunc:.4f}')
            logger.info(f'------------train-loss: {epoch_loss:.4f}')

            # Testing phase on Validation Set
            loss_val = 0
            loss_val_detect = 0
            loss_val_class = 0
            loss_val_kfunc = 0
            tp_count_all = np.zeros((n_classes_out))
            fp_count_all = np.zeros((n_classes_out))
            fn_count_all = np.zeros((n_classes_out))
            test_count_k = 0
            model.eval()
            
            with torch.no_grad():
                with tqdm(test_loader, unit='batch', desc=f"Valid Step [{epoch}]", ascii=' =', dynamic_ncols=True) as batch_loader:
                    for (img, gt_dmap, gt_dots, gt_kmap, __) in batch_loader:
                        ''' 
                            img: input image
                            gt_dmap: ground truth map for cell classes (lymphocytes, epithelial/tumor, stromal) with dilated dots. This can be a binary mask or a density map ( in which case it will be converted to a binary mask)
                            gt_dots: ground truth binary dot map for cell classes (lymphocytes, epithelial/tumor, stromal). 
                            gt_kmap: ground truth k-function map. At each cell center contains the cross k-functions centered at that cell. 
                            img_name: img filename
                        '''
                        gt_kmap /= k_norm_factor # Normalize K functions ground truth
                        # Convert ground truth maps to binary masks (in case they were density maps)
                        gt_dmap = gt_dmap > 0
                        # Get the detection ground truth maps from the classes ground truth maps
                        gt_dmap_all =  gt_dmap.max(dim=1, keepdim=True)[0]
                        gt_dots_all =  gt_dots.max(dim=1, keepdim=True)[0]
                        # Set datatype and move to GPU
                        gt_dmap = gt_dmap.type(torch.FloatTensor)
                        gt_dmap_all = gt_dmap_all.type(torch.FloatTensor)
                        gt_kmap = gt_kmap.type(torch.FloatTensor)

                        # forward Propagation
                        loss_dice_all, loss_dice_class, loss_l1_k, et_all_sig, et_class_sig = model(x=img,
                                                                                                    gt={'gt_dmap_all':gt_dmap_all,
                                                                                                        'gt_dmap':gt_dmap,
                                                                                                        'gt_dmap_subclasses':None,
                                                                                                        'gt_kmap':gt_kmap},
                                                                                                    loss_fns={'criterion_l1_sum':criterion_l1_sum,
                                                                                                              'criterion_sig':criterion_sig,
                                                                                                              'criterion_softmax':criterion_softmax},
                                                                                                    phase='test')

                        loss_dice_all = loss_dice_all.mean()
                        loss_dice_class = loss_dice_class.mean()
                        loss_l1_k = (loss_l1_k.sum() / (gt_dmap_all.clone().sum()*k_func_dim)).item()

                        loss_dice = (lamda_class*loss_dice_class + lamda_detect*loss_dice_all).item()

                        loss = loss_dice
                        if(not math.isnan(loss_l1_k)):
                            loss += loss_l1_k * lamda_k
                            loss_val_kfunc += loss_l1_k
                            test_count_k += 1

                        loss_val += loss
                        loss_val_detect += loss_dice_all.item()
                        loss_val_class += loss_dice_class.item()
                        
                        # Convert ground truth maps and preds to numpy arrays
                        gt_dots = gt_dots.numpy()
                        gt_dots_all = gt_dots_all.numpy()
                        gt_dmap = gt_dmap.numpy()
                        et_all_sig = et_all_sig.cpu().numpy()
                        et_class_sig = et_class_sig.cpu().numpy()
                        
                        # start_time = time.time()
                        # Calculate F-score if epoch >= epoch_start_eval_prec
                        if(epoch >= epoch_start_eval_prec):
                            
                            for k in range(gt_dots_all.shape[0]):
                                # Apply a 0.5 threshold on detection output and convert to binary mask
                                e_hard2 = filters.apply_hysteresis_threshold(et_all_sig[k].squeeze(0), 0.5, 0.5)            
                                e_hard2 = (e_hard2 > 0).astype(np.uint8)
                                e_hard2_all = e_hard2.copy() 

                                e_hard2 = label(e_hard2)
                                counter = Counter(e_hard2.flatten())
                                filtered_counter = {key: value for key, value in counter.items() if value <= 3} # remove objects with less than 3 pixels
                                remove_labels = list(filtered_counter.keys())
                                mask = np.isin(e_hard2, remove_labels)
                                e_hard2[mask] = 0

                                props = regionprops_table(e_hard2, properties=['centroid'])
                                e_hard2 = (e_hard2 > 0).astype(np.uint8)

                                e_dot_all = np.zeros_like(e_hard2)
                                e_dot_all[..., props['centroid-0'].astype(np.int32), props['centroid-1'].astype(np.int32)] = 1


                                label_e_hard2 = label(e_hard2_all)
                                total_g_nuclei = int(gt_dots_all[k].sum())
                                total_e_nuclei = np.unique(label_e_hard2).shape[0] - 1
                                # detection true positives are the number intersections between the ground truth and the detections
                                # Note: if more than one ground truth dot interests, then only one is a TP.
                                tp_count = np.unique(label_e_hard2 * gt_dots_all[k]).shape[0] - 1
                                tp_count_all[-1] += tp_count

                                # False positives is the same but with the part of the image that is not nuclei
                                fp_count_all[-1] += total_e_nuclei - tp_count

                                # False negatives are the part of the GT centers that do not intersect with the detections (292)
                                fn_count_all[-1] += total_g_nuclei - tp_count

                                # Now the same approach but per class
                                for s in range(n_classes):

                                    et_class_argmax = et_class_sig[k].argmax(axis=0)
                                    e_hard2 = (et_class_argmax == s)  
                                    g_dot = gt_dots[k,s,...]

                                    e_dot = (e_hard2 * e_dot_all)
                                    e_centroids = np.array(np.where(e_dot>0)).T
                                    g_centroids = np.array(np.where(g_dot>0)).T
                                    paired, unpaired_true, unpaired_pred = pair_coordinates(g_centroids, e_centroids, 6)
                                    tp_count_all[s] += len(paired)
                                    fp_count_all[s] += len(unpaired_pred)
                                    fn_count_all[s] += len(unpaired_true)

                        batch_loader.set_postfix({"DiceLoss": loss_dice, "K-FuncLoss":loss_l1_k})
                        # start_time = time.time()
            del img, gt_dmap, gt_dots, gt_kmap
            empty_trash()
            logger.info('------------validation-loss-detect: {:.4f}'.format(loss_val_detect/len(test_loader)))
            logger.info('------------validation-loss-class: {:.4f}'.format(loss_val_class/len(test_loader)))
            logger.info('------------validation-loss-kfunc: {:.4f}'.format(loss_val_kfunc/test_count_k))
            logger.info(f'------------validation-loss: {loss_val/len(test_loader):.4f}')
            saved = False

            last_epoch_filepath = os.path.join(checkpoints_save_path, 'mcspat_last.pth')
            torch.save({'desc': model.state_dict(), 
                        'epoch':epoch+1}, 
                        last_epoch_filepath)  
            centroids.dump(os.path.join(checkpoints_save_path, 'centroids_last.npy'))
            

            args.cell_code[len(args.cell_code)] = 'detection'
            precision_all = np.zeros((n_classes_out))
            recall_all = np.zeros((n_classes_out))
            f1_all = np.zeros((n_classes_out))
            if(epoch >= epoch_start_eval_prec):
                count_all = tp_count_all.sum() + fn_count_all.sum()
                for s in range(n_classes_out):
                    if(tp_count_all[s] + fp_count_all[s] == 0):
                        precision_all[s] = 1
                    else:
                        precision_all[s] = tp_count_all[s]/(tp_count_all[s] + fp_count_all[s])
                    if(tp_count_all[s] + fn_count_all[s] == 0):
                        recall_all[s] = 1
                    else:
                        recall_all[s] = tp_count_all[s]/(tp_count_all[s] + fn_count_all[s])
                    if(precision_all[s]+recall_all[s] == 0):
                        f1_all[s] = 0
                    else:
                        f1_all[s] = 2*(precision_all[s] *recall_all[s])/(precision_all[s]+recall_all[s])
                    if s == 0:
                        logger.info(f'------Validation metrics print-out (per class):')
                    logger.info(f'------------class {args.cell_code[s]}: precision_all {precision_all[s]:.4f}, recall_all {recall_all[s]:.4f}, f1_all {f1_all[s]:.4f}')

                logger.info(f'------------mean classes: precision_all {precision_all[:-1].mean():.4f}, recall_all {recall_all[:-1].mean():.4f}, f1_all {f1_all[:-1].mean():.4f}')


            # Check if this is the best epoch so far based on fscore on validation set
            model_save_postfix = ''
            is_best_epoch = False
            if (f1_all.mean() - best_f1_mean >= 0.005):
                model_save_postfix += '_f1'
                best_f1_mean = f1_all.mean()
                best_prec_recall_diff = abs(recall_all.mean()-precision_all.mean())
                is_best_epoch = True
            elif ((abs(f1_all.mean() - best_f1_mean) < 0.005) # a slightly lower f score but smaller gap between precision and recall
                    and abs(recall_all.mean()-precision_all.mean()) < best_prec_recall_diff):
                model_save_postfix += '_pr-diff'
                best_f1_mean = f1_all.mean()
                best_prec_recall_diff = abs(recall_all.mean()-precision_all.mean())
                is_best_epoch = True
            
            # Save checkpoint if it is best so far
            if((saved == False) and (model_save_postfix != '')):
                # print('epoch', epoch, 'saving')
                logger.info('------Saving checkpoint and predictions...')
                new_epoch_filepath = os.path.join(checkpoints_save_path, 'mcspat_epoch_'+str(epoch)+model_save_postfix+".pth")
                torch.save(model.state_dict(), new_epoch_filepath ) # save only if get better error
                centroids.dump(os.path.join(checkpoints_save_path, 'epoch{}_centroids.npy'.format(epoch)))
                saved = True
                logger.info('------Checkpoint and predictions saved.')

                if(is_best_epoch):
                    best_epoch_filepath = new_epoch_filepath
                    best_epoch = epoch     
            # Adam optimizer needs resetting to avoid parameters learning rates dying
            sys.stdout.flush();           
    logger.info('------Training complete.')