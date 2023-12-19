import os
import sys;

print(os.environ["CONDA_PREFIX"])
print("BEFORE EDITING PYTHONPATH\n", sys.path, "\n")
sys.path.remove('/software.el7/software/SciPy-bundle/2021.05-foss-2021a/lib/python3.9/site-packages')
sys.path.remove('/software.el7/software/GDAL/3.3.0-foss-2021a/lib/python3.9/site-packages')
# sys.path.remove('/software.el7/software/pybind11/2.6.2-GCCcore-10.3.0/lib/python3.9/site-packages')
sys.path.append('/storage/homefs/jg23p152/.conda/envs/hovernet/lib/python3.9/site-packages')
print("AFTER EDITING PYTHONPATH\n", sys.path, "\n")
import numpy as np
import skimage.io as io
from skimage.measure import label
import glob
from natsort import natsorted
import argparse
sys.path.append("/storage/homefs/jg23p152/project/MCSpatNet")
from spatial_analysis_utils_v2_sh import *
import logging
from tqdm import tqdm
from utils import create_logger, get_kmaps_extract_args, divide_list
# Calculates the cross K-function at each cell and propagate the same values to all the pixels in the cell connected components in the ground truth dilated dot map or binary mask.


def get_npy_files(root_dir):
    # Use glob to find all .npy files in root_dir and its subdirectories
    npy_files = glob.glob(os.path.join(root_dir, '**', '*.npy'), recursive=True)
    return natsorted(npy_files)

if __name__ == "__main__":

    args = get_kmaps_extract_args()
    logger = create_logger()

    slurm_job = 'slurm job array' if os.environ.get('SLURM_JOB_ID') else 'local machine'
    slurm_array_task_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    slurm_array_job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    logger.info(f'Script started. Running in {slurm_job}. Task ID: {slurm_array_job_id+1} out of {slurm_array_task_count}')

    # file_list = get_npy_files(args.dataset_path)

    # Configure K function
    do_k_correction=True
    n_classes = args.nr_types
    r_step = 15 # r stands for radius
    r_range = range(0, 100, r_step)
    r_list = [*r_range]
    r_classes = len(r_range)
    r_classes_all = r_classes * (n_classes)

    # os.makedirs(out_dir, exist_ok=True)

    img_path_list = get_npy_files(args.dataset_path)
    img_path_list = divide_list(img_path_list, slurm_array_task_count)[slurm_array_job_id]
    logger.info(f'Processing {len(img_path_list)} images...')

    pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    pbarx = tqdm(total=len(img_path_list), bar_format=pbar_format, ascii=True, position=0)
    # Calculate the cross K-function at each cell and propagate the same values to all the pixels in the cell connected components in the ground truth dilated dot map or binary mask.
    for img_path in img_path_list:

        array_all = np.load(img_path, allow_pickle=True)

        if array_all.shape[-1] == 32:
            pbarx.update()
            continue

        logger.info(f'Processing {img_path}')
        array_all = array_all[...,:11]
        img = array_all[...,:3]
        gt_dots = array_all[...,5:8]
        gt_dmap = array_all[...,8:11]

        # Load the ground truth dot maps and binary masks
        # logger.info(f'img {img_path}')
        # img_name = os.path.basename(img_path)
        # gt_path = os.path.join(gt_dir,img_name.replace('.png','_gt_dots.npy'));
        # gt_dots=np.load(gt_path, allow_pickle=True)[:,:,1:].squeeze()
        # gt_dmap_path = os.path.join(gt_dir,img_name.replace('.png','.npy'));
        # gt_dmap=np.load(gt_dmap_path, allow_pickle=True)[:,:,1:].squeeze()
        gt_dots_all = gt_dots.max(-1)
        gt_dmap = gt_dmap > 0
        gt_dmap_all = gt_dmap.max(-1)
        gt_dmap_all_comp = label(gt_dmap_all) # label all connected components to make it easy to propagate the k function to the cell's pixels
        # gt_kmap_out_path = os.path.join(out_dir,img_name.replace('.png','_gt_kmap.npy')); # output filepath
        k_area = gt_dots.shape[0]*gt_dots.shape[1]

        # cells_y, cells_x arrays to hold cell dot coordinates, cells_mark array to hold cells classes index
        # First entry is reserved to be used as the center cell, which will have its own identifier, when computing the K-function, and so it will change with center cell changing
        cells_y=[0]
        cells_x=[0]
        cells_mark=['1000']

        # load the cells' coordinates and classes into cells_y, cells_x, and cells_mark
        for c in range(n_classes):
            c_points = np.where(gt_dots[:,:, c] > 0)
            if(len(c_points[0])>0):
                cells_y = np.concatenate((cells_y, c_points[0]))
                cells_x = np.concatenate((cells_x, c_points[1]))
                cells_mark =  cells_mark + [str(c+1)]*len(c_points[0])

        # initialize the kmap array
        gt_kmap = np.zeros((gt_dots.shape[0],gt_dots.shape[1],r_classes_all))

        # c_points are the center points, which are all the cells
        c_points = np.where(gt_dots_all > 0)
        if(len(c_points[0]) != 0):

        # '''
        #     Loop over each cell in c_points
        #         Set the first entry in cells_y and cells_x to the current cell coordinates
        #         Compute the K-function with respect to each of the other classes.
        #         In the kmap, set all the pixels in the current cell connected component to the calculated k-function
        # '''
            for ci in range(len(c_points[0])):
                cy = c_points[0][ci]
                cx = c_points[1][ci]
                comp_indx = gt_dmap_all_comp[cy,cx]
                cells_y[0] = cy
                cells_x[0] = cx
                cells_ppp = ppp(cells_x, cells_y, cells_mark) # the point pattern in R object
                k_indx = 0
                c_k_func = np.zeros(r_classes_all)
                for s2 in range(n_classes):
                    if(gt_dots[:,:, s2].sum() > 0):
                        if(do_k_correction):
                            r_Kcross, K_val_samp = Kcross(cells_ppp, i='1000', j=str(s2+1), correction='iso', plot=False, r=r_range)
                        else:
                            r_Kcross, K_val_samp = Kcross(cells_ppp, i='1000', j=str(s2+1), correction='none', plot=False, r=r_range)
                        c_k_func[k_indx:k_indx+r_classes] = K_val_samp/k_area * gt_dots[:,:, s2].sum()
                    else:
                        c_k_func[k_indx:k_indx+r_classes] = 0
                    k_indx += r_classes
                gt_kmap[gt_dmap_all_comp == comp_indx] = c_k_func
        array_new = np.concatenate((array_all, gt_kmap), axis=-1)
        logger.info(f'Saving array of shape {array_new.shape}')
        # ADDS THE 21 KMAPS TO THE ARRAY
        np.save(img_path, array_new.astype(np.float16))
        pbarx.update()
    pbarx.close()
    logger.info('Done!')

