import os
import numpy as np
from skimage import io;
import cv2 ;
import sys;
from skimage.measure import label, moments
from skimage import filters
from tqdm import tqdm as tqdm
import torch
import torch.nn as nn
import glob
from natsort import natsorted
from model_arch_custom import UnetVggMultihead_custom as UnetVggMultihead
from my_dataloader_custom import CellsDataset
import argparse
from scipy.io import savemat

def get_reconstructed_simple(crop_paths, original_size):
    """To be used for all pngs

    Args:
        crop_paths (list): list of paths to the crops
        original_size (tuple): size of the original image
        x_to_drop (int): pixels to drop at each side of the image in x direction
        y_to_drop (int): pixels to drop at each side of the image in y direction
        n_c (int): number of channels of image

    Returns:
        np.ndarray: reconstructed image
    """
    
    new_size = int(os.path.basename(crop_paths[-1]).split('_')[4]), int(os.path.basename(crop_paths[-1]).split('_')[7])
    x_to_drop = (new_size[0] - original_size[0])//2 if new_size[0] - original_size[0] > 0 else 0
    y_to_drop = (new_size[1] - original_size[1])//2 if new_size[1] - original_size[1] > 0 else 0

    if crop_paths[0].endswith('.npy'):
        tmp = np.load(crop_paths[0], allow_pickle=True)
        # n_c = np.load(crop_paths[4], allow_pickle=True).shape[-1] if len(np.load(crop_paths[0], allow_pickle=True).shape) == 3 else 1
    else:
        tmp = cv2.imread(crop_paths[0])
    n_c = tmp.shape[-1] if len(tmp.shape) == 3 else 0
    
    reconstructed = np.zeros((original_size[0]+x_to_drop*2, original_size[1]+y_to_drop*2, n_c)) if n_c > 0 else np.zeros((original_size[0]+x_to_drop*2, original_size[1]+y_to_drop*2))
    
    for sample in crop_paths:
        # Extract coordinates from the filename
        file_name = os.path.basename(sample)
        coords_x = [int(file_name.split('_')[3]), int(file_name.split('_')[4])]
        coords_y = [int(file_name.split('_')[6]), int(file_name.split('_')[7])]

        # Load the crop
        crop = np.load(sample, allow_pickle=True) if sample.endswith('.npy') else cv2.imread(sample)

        # Add the crop to the corresponding location in the reconstructed image
        reconstructed[coords_x[0]:coords_x[1], coords_y[0]:coords_y[1]] = crop
    reconstructed = reconstructed[x_to_drop:-x_to_drop, y_to_drop:-y_to_drop]
    return reconstructed

def get_infer_args():
    parser= argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/storage/homefs/jg23p152/project', help='The root directory.')
    parser.add_argument('--out_dir', type=str, help='The name of the folder that will be created to save the inference output.')
    parser.add_argument('--model_path', type=str, default=None, help='path of a previous checkpoint to continue training')
    parser.add_argument('--train_data_root', type=str, default='/storage/homefs/jg23p152/project/MCSpatNet/datasets/Lizard', help='path to training data')
    parser.add_argument('--test_data_root', type=str, default='/storage/homefs/jg23p152/project/MCSpatNet/datasets/Lizard', help='path to testing data')
    parser.add_argument('--test_split_filepath', type=str, default='/storage/homefs/jg23p152/project/Data/Split_files/cv0_val_files.json', help='path to testing data split file')

    parser.add_argument('--cropped_data', action='store_true', help='Set if using with pre-cropped data')
    parser.add_argument('--batch_size', type=int, default=40, help='batch size')
    parser.add_argument('--cell_code', type=str, default='1:others,2:epit_healthy,3:epit_maligant', help='dictionary of cell classes')

    parser.add_argument('--seed', type=int, default=123, help='random seed')

    args = parser.parse_args()
    args.cell_code = {int(s.split(':')[0]):s.split(':')[1] for s in args.cell_code.split(',')}

    return args

if __name__=="__main__":
    args = get_infer_args()
    checkpoints_root_dir = args.root_dir # The root directory for all training output.

    # checkpoints_folder_name = 'mcspatnet_consep_1' # The name of the current training output folder under <checkpoints_root_dir>.
    out_dir = args.out_dir # The name of the folder that will be created to save the inference output.
    visualize=True # whether to output a visualization of the prediction
    test_data_root = None
    test_split_filepath = args.test_split_filepath

    model_path = args.model_path
    # Initializations
    #2: Malignant Epit: red
    #1: Healthy Epit: green
    #0: Other: black
    color_set = {0:(0,0,0), 1:(0,255,0), 2:(0,0,255)} 

    # model checkpoint and output configuration parameters

    os.makedirs(out_dir, exist_ok=True)

    # data configuration parameters
    # test_image_root = os.path.join(test_data_root, 'images')
    # test_dmap_root = os.path.join(test_data_root, 'gt_custom')
    # test_dots_root = os.path.join(test_data_root, 'gt_custom')

    # Model configuration parameters
    gt_multiplier = 1    
    gpu_or_cpu='cuda' if torch.cuda.is_available() else 'cpu'# use cuda or cpu
    dropout_prob = 0
    initial_pad = 126
    interpolate = 'False'
    conv_init = 'he'
    n_classes = 3
    n_classes_out = n_classes + 1
    class_indx = '1,2,3'
    class_weights = np.array([1,1,1]) 
    n_clusters = 5
    n_classes2 = n_clusters * (n_classes)

    r_step = 15
    r_range = range(0, 100, r_step)
    r_arr = np.array([*r_range])
    r_classes = len(r_range)
    r_classes_all = r_classes * (n_classes )

    thresh_low = 0.5
    thresh_high = 0.5
    size_thresh = 5

    slurm_job = 'slurm job array' if os.environ.get('SLURM_JOB_ID') else 'local machine'
    slurm_array_task_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    slurm_array_job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    logger.info(f'Script started. Running in {slurm_job}. Task ID: {slurm_array_job_id+1} out of {slurm_array_task_count}')

    file_dict_list = parse_json_file(args.test_split_filepath)
    file_dict_list = divide_list(file_dict_list, slurm_array_task_count)[slurm_array_job_id]
    img_ids = natsorted([file_dict['img_id'] for file_dict in file_dict_list])
    args.test_split_filepath = os.path.join(out_dir, f"tmp_lst_{os.environ.get('SLURM_JOB_ID')}_{slurm_array_job_id}.json")
    save_json(os.path.dirname(args.test_split_filepath),os.path.basename(args.test_split_filepath), file_dict_list)
    logger.info(f'Number of images: {len(img_ids)}')
    logger.info(f'Image IDs: {img_ids}')

    device=torch.device(gpu_or_cpu)
    # model=nn.DataParallel(UnetVggMultihead(kwargs={'dropout_prob':dropout_prob, 'initial_pad':initial_pad, 'interpolate':interpolate, 'conv_init':conv_init, 'n_classes':n_classes, 'n_channels':3, 'n_heads':4, 'head_classes':[1,n_classes,n_classes2, r_classes_all]}))
    model=UnetVggMultihead(kwargs={'dropout_prob':dropout_prob, 'initial_pad':initial_pad, 'interpolate':interpolate, 'conv_init':conv_init, 'n_classes':n_classes, 'n_channels':3, 'n_heads':4, 'head_classes':[1,n_classes,n_classes2, r_classes_all], 'use_dice_loss':True, 'use_ce_loss':False})
    # model.to(device)
    criterion_sig = nn.Sigmoid() # initialize sigmoid layer
    criterion_softmax = nn.Softmax(dim=1) # initialize sigmoid layer
    test_dataset=CellsDataset(args.root_dir,class_indx, split_filepath=args.test_split_filepath, phase='test', fixed_size=-1, max_scale=16)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False)

    logger.info(f'thresh {thresh_low}, {thresh_high}')

    # Load model
    logger.info('test epoch :  Best' )
    

    sys.stdout.flush();
    if 'pretrained_models' in model_path:
        model.load_state_dict(torch.load(model_path,map_location=torch.device(gpu_or_cpu)), strict=True);
        model = nn.DataParallel(model)
    else:
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path,map_location=torch.device(gpu_or_cpu)), strict=True);
    model.to(device)
    model.eval()
    sample_paths = []
    
    with torch.no_grad():
        with tqdm(test_loader, unit='batch', desc="Infer", dynamic_ncols=True) as iterator:
            for imgs, gt_dmap, gt_dmap_all, gt_dots, gt_dots_all, __, img_names in iterator:
            
                sys.stdout.flush();

                # Forward Propagation
                loss_dice_all, loss_dice_class, loss_l1_k, et_all_sig, et_class_sig = model(x=imgs,
                                                                                            gt={'gt_dmap_all':gt_dmap_all.type(torch.FloatTensor),
                                                                                                'gt_dmap':gt_dmap.type(torch.FloatTensor),
                                                                                                'gt_dmap_subclasses':None,
                                                                                                'gt_kmap':torch.zeros((imgs.shape[0], 21, imgs.shape[2], imgs.shape[3]))},
                                                                                            loss_fns={'criterion_l1_sum':nn.L1Loss(reduction='sum'),
                                                                                                        'criterion_sig':criterion_sig,
                                                                                                        'criterion_softmax':criterion_softmax},
                                                                                            phase='test')
                imgs = imgs[...,2:-2,2:-2]

                gt_dmap_all = gt_dmap_all[...,2:-2,2:-2]
                gt_dmap = gt_dmap[...,2:-2,2:-2]
                gt_dots = gt_dots[...,2:-2,2:-2]
                gt_dots_all = gt_dots_all[...,2:-2,2:-2]

                et_all_sig = et_all_sig[...,2:-2,2:-2]
                et_class_sig = et_class_sig[...,2:-2,2:-2]
                
                for j in range(len(imgs)):
                    img = imgs[j].cpu().permute(1,2,0).numpy()*255

                    et_all_sig_single = et_all_sig[j].cpu().permute(1,2,0).numpy()
                    et_class_sig_single = et_class_sig[j].cpu().permute(1,2,0).numpy()

                    gt_dots_single = gt_dots[j].cpu().permute(1,2,0).numpy()
                    gt_dots_all_single = gt_dots_all[j].cpu().permute(1,2,0).numpy()
                    gt_dmap_single = gt_dmap_all[j].cpu().permute(1,2,0).numpy()


                    img_centers_all = img.copy()
                    img_centers_all_gt = img.copy()

                    img_centers_all_all = img.copy()
                    img_centers_all_all_gt = img.copy()

                    img_name = os.path.normpath(img_names[j]).split(os.path.sep)[-2:]
                    img_name = os.path.join(img_name[0], img_name[1])
                    sample_path = os.path.join(out_dir, os.path.dirname(img_name))
                    os.makedirs(sample_path, exist_ok=True)
                    sample_paths.append(sample_path)

                    g_count = gt_dots_all_single.sum()
                    # Get connected components in the prediction and apply a small size threshold
                    e_hard = filters.apply_hysteresis_threshold(et_all_sig_single.squeeze(), thresh_low, thresh_high)
                    e_hard2 = (e_hard > 0).astype(np.uint8)
                    comp_mask = label(e_hard2)
                    e_count = comp_mask.max()
                    s_count=0
                    if(size_thresh > 0):
                        for c in range(1,comp_mask.max()+1):
                            s = (comp_mask == c).sum()
                            if(s < size_thresh):
                                e_count -=1
                                s_count +=1
                                e_hard2[comp_mask == c] = 0
                    e_hard2_all = e_hard2.copy()

                    # Get centers of connected components in the prediction
                    e_dot = np.zeros((img.shape[0], img.shape[1]))
                    e_dot_vis = np.zeros((img.shape[0], img.shape[1]))
                    contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for idx in range(len(contours)):
                        contour_i = contours[idx]
                        M = cv2.moments(contour_i)
                        if(M['m00'] == 0):
                            continue;
                        cx = round(M['m10'] / M['m00'])
                        cy = round(M['m01'] / M['m00'])
                        e_dot_vis[cy-1:cy+1, cx-1:cx+1] = 1
                        e_dot[min(cy, e_dot.shape[0]-1), min(cx, e_dot.shape[1]-1)] = 1
                        img_centers_all_all[cy-3:cy+3, cx-3:cx+3,:] = (0,0,0)
                    e_dot_all = e_dot.copy()
                    gt_centers = np.where(gt_dots_all_single > 0)
                    for idx in range(len(gt_centers[0])):
                        cx = gt_centers[1][idx]
                        cy = gt_centers[0][idx]
                        img_centers_all_all_gt[cy-3:cy+3, cx-3:cx+3,:] = (0,0,0)

                    e_dot.astype(np.uint8).dump(
                        os.path.join(out_dir, img_name.replace('.npy',  '_centers' + '_all' + '.npy')))
                    if visualize:
                        #io.imsave(os.path.join(out_dir, img_name.replace('.npy','_centers'+'_allcells' +'.png')), (e_dot_vis*255).astype(np.uint8))
                        io.imsave(os.path.join(out_dir, img_name.replace('.npy','_centers'+'_det' +'_overlay.png')), (img_centers_all_all).astype(np.uint8))
                        #io.imsave(os.path.join(out_dir, img_name.replace('.npy','_allcells' +'_hard.png')), (e_hard2*255).astype(np.uint8))

                    # end: eval detection all

                    # begin: eval classification
                    et_class_argmax = et_class_sig_single.argmax(axis=-1)
                    e_hard2_all = e_hard2.copy()

                    for s in range(n_classes-1):
                        g_count = gt_dots_single[...,s].sum() # gt_dots does not have background class as 0
                        e_hard2 = (et_class_argmax == s+1) # et_class has background class as 0

                        # Filter the predicted detection dot map by the current class predictions
                        e_dot = e_hard2 * e_dot_all
                        e_count = e_dot.sum()

                        g_dot = gt_dots_single[...,s]
                        e_dot_vis = np.zeros(g_dot.shape)
                        e_dots_tuple = np.where(e_dot > 0)
                        for idx in range(len(e_dots_tuple[0])):
                            cy=e_dots_tuple[0][idx]
                            cx=e_dots_tuple[1][idx]
                            img_centers_all[cy-3:cy+3, cx-3:cx+3,:] = color_set[s]


                        gt_centers = np.where(g_dot > 0)
                        for idx in range(len(gt_centers[0])):
                            cx = gt_centers[1][idx]
                            cy = gt_centers[0][idx]
                            img_centers_all_gt[cy-3:cy+3, cx-3:cx+3,:] = color_set[s]

                        e_dot.astype(np.uint8).dump(os.path.join(out_dir, img_name.replace('.npy', '_centers' + '_s' + str(s) + '.npy')))
                        #if(visualize):
                        #    io.imsave(os.path.join(out_dir, img_name.replace('.npy','_likelihood_s'+ str(s)+'.png')), (et_class_sig.squeeze()[s]*255).astype(np.uint8));
                    # end: eval classification


                    et_class_sig_single.astype(np.float16).dump(
                        os.path.join(out_dir, img_name.replace('.npy', '_likelihood_class' + '.npy')))
                    et_all_sig_single.astype(np.float16).dump(
                        os.path.join(out_dir, img_name.replace('.npy', '_likelihood_all' + '.npy')))
                    gt_dots_single.astype(np.uint8).dump(
                        os.path.join(out_dir, img_name.replace('.npy', '_gt_dots_class' + '.npy')))
                    gt_dots_all_single.astype(np.uint8).dump(
                        os.path.join(out_dir, img_name.replace('.npy', '_gt_dots_all' + '.npy')))
                    if(visualize):
                        io.imsave(os.path.join(out_dir, img_name.replace('.npy','_centers'+ '_class_overlay' +'.png')), (img_centers_all).astype(np.uint8))
                        io.imsave(os.path.join(out_dir, img_name.replace('.npy','_gt_centers'+'_class_overlay'+'.png')), (img_centers_all_gt).astype(np.uint8))
                        # io.imsave(os.path.join(out_dir, img_name), (img).astype(np.uint8))
                        #io.imsave(os.path.join(out_dir, img_name.replace('.png','_likelihood_all'+'.png')), (et_all_sig.squeeze()*255).astype(np.uint8));
    
    paths = [x[0] for x in os.walk(args.out_dir) if os.path.basename(x[0]) != 'mat'][1:]
    for path in paths:
        name = os.path.basename(path)
        for file_end in ['centers_all.npy', '_centers_det_overlay.png', '_centers_s0.npy', '_centers_s1.npy', '_centers_s2.npy','_gt_centers_class_overlay.png', '_centers_class_overlay.png', '_likelihood_class.npy', '_likelihood_all.npy', '_gt_dots_class.npy', '_gt_dots_all.npy']:
            crop_paths = natsorted(glob.glob(os.path.join(path, '*'+file_end), recursive=True))
            original_size = cv2.imread(glob.glob(os.path.join('/storage/homefs/jg23p152/project/Data', '**', f'*{name}.png'), recursive=True)[0]).shape[:-1]
            crop_paths = [file for file in crop_paths if 'reconstructed' not in file]
            reconstructed = get_reconstructed_simple(crop_paths, original_size)
            
            if '.png' in file_end:
                io.imsave(os.path.join(path, 'reconstructed_'+file_end), reconstructed.astype(np.uint8))
            else:
                reconstructed.astype(np.uint8).dump(os.path.join(path, 'reconstructed_'+file_end))
            for crop_path in crop_paths:
                os.remove(crop_path)
    
    os.makedirs(os.path.join(args.out_dir, 'mat'), exist_ok=True)
    paths = glob.glob(os.path.join(args.out_dir, '**', 'reconstructed__centers_s*.npy'), recursive=True)
    centers = []
    names = list(set([os.path.normpath(path).lstrip(os.path.sep).split(os.path.sep)[-2] for path in paths]))
    labels = []
    for name in names:
        paths_tmp = [path for path in paths if name+'/' in path]
        mat_file ={}
        for i,path in enumerate(paths_tmp):
            arr = np.load(path, allow_pickle=True)
            centers_temp_x, centers_temp_y = np.where(arr != 0)
            labels.extend([i]*centers_temp_x.shape[0])
            centers.append(np.stack((centers_temp_x, centers_temp_y), axis=1))
        centers = np.concatenate(centers, axis=0)
        labels = np.array(labels)
        mat_file['inst_centroid'] = centers
        mat_file['inst_type'] = labels
        savemat(os.path.join(args.out_dir, 'mat', name+'.mat'), mat_file)
    
