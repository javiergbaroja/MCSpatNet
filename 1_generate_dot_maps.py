import numpy as np
import glob
import os
import sys
import skimage.io as io
from scipy import ndimage
import scipy.io as sio
import cv2
import scipy
import argparse
sys.path.append("/storage/homefs/jg23p152/project")
from hover_net.misc.utils import parse_json_file
from tqdm import tqdm
from utils import get_gt_extract_args, create_logger, divide_list
from natsort import natsorted

class LabelHandler():
    def __init__(self, 
                 dataset:str,
                 nr_types:int,
                 n_grouped_class_channels:int,
                 class_group_mapping_dict:dict,
                 color_set:dict,
                 img_scale:float=0.5,
                 subtyping:bool=True,
                 type2subtype:int=2,
                 remove_duplicates:bool=False,
                 save_vis:bool=True,
                 save_as_crops:bool=False,
                 window_size:int=540,
                 step_size:int=164) -> None:
        
        
        self.dataset = dataset
        self.nr_types = nr_types
        self.n_grouped_class_channels = n_grouped_class_channels
        self.class_group_mapping_dict = class_group_mapping_dict
        self.color_set = color_set
        self.img_scale = img_scale
        self.subtyping = subtyping
        self.type2subtype = type2subtype
        self.remove_duplicates = remove_duplicates
        self.save_vis = save_vis
        self.save_as_crops = save_as_crops
        self.root_dir = None
        self.out_img_dir = None
        self.out_gt_dir = None
        self.win_size=[window_size,window_size] 
        self.step_size=[step_size,step_size]
        

        if dataset.lower() == "tcga":
            self.centroid_key = "inst_centroid"
            self.class_key = "inst_type"
        elif dataset.lower() in ['lizard', 'pannuke']:
            self.centroid_key = "centroid"
            self.class_key = "class"
        else:
            raise ValueError("dataset must be one of ['tcga', 'lizard', 'pannuke']")
    
    def _set_root_dir(self, root_dir:str):
        self.root_dir = root_dir
        return
    
    def _get_coords_from_filename(self, filename:str):
        return [[int(os.path.splitext(filename)[0].split("x")[1].split("_")[1]), 
                 int(os.path.splitext(filename)[0].split("x")[1].split("_")[2])], 
                [int(os.path.splitext(filename)[0].split("y")[1].split("_")[1]), 
                 int(os.path.splitext(filename)[0].split("y")[1].split("_")[2])]]

    def _gaussian_filter_density(self, img, points, point_class_map, out_filepath, start_y=0, start_x=0, end_y=-1, end_x=-1, save:bool=True):
        '''
            Build a KD-tree from the points, and for each point get the nearest neighbor point.
            The default Guassian width is 9.
            Sigma is adaptively selected to be min(nearest neighbor distance*0.125, 2) and truncate at 2*sigma.
            After generation of each point Gaussian, it is normalized and added to the final density map.
            A visualization of the generated maps is saved in <slide_name>_<img_name>.png and <slide_name>_<img_name>_binary.png
        '''
        img_shape = [img.shape[0], img.shape[1]]
        logger.info(f"Shape of current image: {img_shape}. Totally need generate {len(points)} gaussian kernels.")
        density = np.zeros(img_shape, dtype=np.float32)
        density_class = np.zeros((img.shape[0], img.shape[1], point_class_map.shape[2]), dtype=np.float32)
        if (end_y <= 0):
            end_y = img.shape[0]
        if (end_x <= 0):
            end_x = img.shape[1]
        gt_count = len(points)
        if gt_count == 0:
            return density
        leafsize = 2048
        # build kdtree
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(points, k=2)
        logger.info('generate density...')

        max_sigma = 2;  # kernel size = 4, kernel_width=9

        for i, pt in enumerate(points):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            if (pt[1] < start_y or pt[0] < start_x or pt[1] >= end_y or pt[0] >= end_x):
                continue
            pt[1] -= start_y
            pt[0] -= start_x
            if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
                pt2d[int(pt[1]), int(pt[0])] = 1.
            else:
                continue
            if gt_count > 1:
                sigma = (distances[i][1]) * 0.125
                sigma = min(max_sigma, sigma)
            else:
                sigma = max_sigma;

            kernel_size = min(max_sigma * 2, int(2 * sigma + 0.5))
            sigma = kernel_size / 2
            kernel_width = kernel_size * 2 + 1
            # if(kernel_width < 9):
            #     print('i',i)
            #     print('distances',distances.shape)
            #     print('kernel_width',kernel_width)
            pnt_density = scipy.ndimage.gaussian_filter(pt2d, sigma, mode='constant', truncate=2)
            pnt_density /= pnt_density.sum()
            density += pnt_density
            class_indx = point_class_map[int(pt[1]), int(pt[0])].argmax()
            density_class[:, :, class_indx] = density_class[:, :, class_indx] + pnt_density
        if save:

            #density_class.astype(np.float16).dump(out_filepath)
            #density.astype(np.float16).dump(os.path.splitext(out_filepath)[0] + '_all.npy')
            (density_class > 0).astype(np.uint8).dump(out_filepath)
            (density > 0).astype(np.uint8).dump(os.path.splitext(out_filepath)[0] + '_all.npy')
            #io.imsave(out_filepath.replace('.npy', '.png'), (density / density.max() * 255).astype(np.uint8))
            io.imsave(out_filepath.replace('.npy', '_binary.png'), ((density > 0) * 255).astype(np.uint8))
            for s in range(1, density_class.shape[-1]):
                io.imsave(out_filepath.replace('.npy', '_s' + str(s) + '_binary.png'),
                        ((density_class[:, :, s] > 0) * 255).astype(np.uint8))
        logger.info('done.')
        return (density > 0).astype(np.uint8),  (density_class > 0).astype(np.uint8)
    
    def _create_out_dirs(self, out_root_dir:str):

        self.out_img_dir = os.path.join(out_root_dir, 'images')
        self.out_gt_dir = os.path.join(out_root_dir, 'gt_custom')

        os.makedirs(out_root_dir, exist_ok=True)
        os.makedirs(self.out_img_dir, exist_ok=True)
        os.makedirs(self.out_gt_dir, exist_ok=True)

        return 
    
    def _scale_img_and_centroids(self, img, centroids):
        
        if self.img_scale != 1:
            img = cv2.resize(img, (int(img.shape[1] * self.img_scale + 0.5), int(img.shape[0] * self.img_scale + 0.5)))
            centroids = (centroids * self.img_scale).astype(int)

            # Make sure coordinates are within limits after scaling image
            # print('centroids',centroids.shape)
            # print('class_types',class_types.shape)

            centroids[(np.where(centroids[:, 1] >= img.shape[0]), 1)] = img.shape[0] - 1
            centroids[(np.where(centroids[:, 0] >= img.shape[1]), 0)] = img.shape[1] - 1

        return img, centroids.astype(int)
    
    
    
    def _create_grouped_gt(self, img, mat):

        def __per_class_mapper(img:np.ndarray, centroids:np.ndarray, class_types:np.ndarray):
            negative = np.where(centroids < 0)
            centroids = np.delete(centroids, negative[0], axis=0)
            class_types = np.delete(class_types, negative[0], axis=0)

            patch_label_arr_dots = np.zeros((img.shape[0], img.shape[1], self.nr_types), dtype=np.uint8)

            # Generated of ground truth classification dot annotation map
            for dot_class in range(1, self.nr_types):
                patch_label_arr = np.zeros((img.shape[0], img.shape[1]))
                positions = np.where(class_types == dot_class)[0]
                patch_label_arr[(centroids[positions][:, 1],
                                 centroids[positions][:, 0])] = 1
                patch_label_arr_dots[:, :, dot_class] = patch_label_arr
                # patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((5, 5)), mode='constant', cval=0.0)
                # img_scaled[np.where(patch_label_arr > 0)] = color_set[dot_class]
            return patch_label_arr_dots
        
        def __label_grouper(patch_label_arr_dots:np.ndarray):
            if self.n_grouped_class_channels < patch_label_arr_dots.shape[-1]:

                patch_label_arr_dots_grouped = np.zeros((patch_label_arr_dots.shape[0], patch_label_arr_dots.shape[1], self.n_grouped_class_channels), dtype=np.uint8)
                for class_id, map_class_lst in self.class_group_mapping_dict.items():
                    patch_label_arr = patch_label_arr_dots[:, :, map_class_lst].sum(axis=-1)
                    patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((9, 9)), mode='constant', cval=0.0)
                    # img3[np.where(patch_label_arr > 0)] = self.color_set[class_id]
                    patch_label_arr_dots_grouped[:, :, class_id] = patch_label_arr_dots[:, :, map_class_lst].sum(axis=-1)
                return patch_label_arr_dots_grouped
            else:
                return patch_label_arr_dots

        # scaling
        img_scaled, centroids = self._scale_img_and_centroids(img, mat[self.centroid_key])
        # centroids = (mat[self.centroid_key] * self.img_scale).astype(int)
        class_types = mat[self.class_key].squeeze()
        patch_label_arr_dots = __per_class_mapper(img_scaled, centroids, class_types)
        patch_label_arr_dots = __label_grouper(patch_label_arr_dots)
        # Visualize of all cell types overlaid dots on img


        return img_scaled, self._remove_duplicates(patch_label_arr_dots)

    def _remove_duplicates(self, patch_label_arr_dots):
        # Remove duplicate dots
        if self.remove_duplicates:
            for c in range(patch_label_arr_dots.shape[-1]):
                tmp = ndimage.convolve(patch_label_arr_dots[:, :, c], np.ones((5, 5)), mode='constant', cval=0.0)
                duplicate_points = np.where(tmp > 1)
                while (len(duplicate_points[0]) > 0):
                    y = duplicate_points[0][0]
                    x = duplicate_points[1][0]
                    patch_label_arr_dots[max(0, y - 2):min(patch_label_arr_dots.shape[0] - 1, y + 3),
                    max(0, x - 2):min(patch_label_arr_dots.shape[1] - 1, x + 3), c] = 0
                    patch_label_arr_dots[y, x, c] = 1
                    tmp = ndimage.convolve(patch_label_arr_dots[:, :, c], np.ones((5, 5)), mode='constant',
                                            cval=0.0)
                    duplicate_points = np.where(tmp > 1)
        return patch_label_arr_dots
    
    
    def _save_visualization(self, img_scaled, patch_label_arr_dots, img_name):
        io.imsave(os.path.join(self.out_img_dir, img_name +'.png'), img_scaled)
        for dot_class in range(1, patch_label_arr_dots.shape[-1]):
            logger.info(f'dot_class {dot_class}')
            logger.info(f'patch_label_arr_dots[:,:,dot_class] {patch_label_arr_dots[:, :, dot_class].sum()}')
            patch_label_arr = patch_label_arr_dots[:, :, dot_class].astype(int)
            patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((5, 5)), mode='constant', cval=0.0)
            img_scaled[np.where(patch_label_arr > 0)] = self.color_set[dot_class]
        io.imsave(os.path.join(self.out_gt_dir, img_name + '_img_with_dots.jpg'), img_scaled)


    def _generate_gaussian_maps(self, img_scaled:np.ndarray, patch_label_arr_dots:np.ndarray, img_name:str, save:bool=False):
        out_gt_dmap_filepath = os.path.join(self.out_gt_dir, img_name  + '.npy')
        mat_s_points = np.where(patch_label_arr_dots > 0)
        points = np.zeros((len(mat_s_points[0]), 2))
        logger.info(points.shape)
        points[:, 0] = mat_s_points[1]
        points[:, 1] = mat_s_points[0]
        patch_label_arr_dots_custom_all, patch_label_arr_dots_custom = self._gaussian_filter_density(img_scaled, points,
                                                                                                patch_label_arr_dots,
                                                                                                out_gt_dmap_filepath,
                                                                                                save=save)
        return patch_label_arr_dots_custom_all, patch_label_arr_dots_custom
    
    def _load_from_file_dict(self, file:dict):
        def _apply_cell_subtyping(mat:dict, file:dict):
            mat[self.class_key] = mat[self.class_key].squeeze()

            if self.dataset.lower() in ['lizard', 'pannuke']:
                mat[self.class_key][mat[self.class_key] != self.type2subtype] = 1
                if file['malignant'] == 1:
                    mat[self.class_key][mat[self.class_key] == self.type2subtype] = 3
                elif file['malignant'] == 0:
                    mat[self.class_key][mat[self.class_key] == self.type2subtype] = 2
                else:
                    raise ValueError("malignant must be 0 or 1 for lizard or pannuke datasets")

            elif self.dataset.lower() == 'tcga': # for TCGA labeling goes as type 1 for healthy, 2 for malignant and 0 for other types
                mat[self.class_key] += 1
            
            return mat

        img_filepath = os.path.join(self.root_dir, file['img_file'])
        mat_filepath = os.path.join(self.root_dir, file['mat_file'])
        img_name = file['img_id']
        logger.info(f'img_filepath {img_filepath}')

        # read img & mat files
        img = io.imread(img_filepath)[:, :, :3]
        mat = sio.loadmat(mat_filepath)

        if self.subtyping:
            mat = _apply_cell_subtyping(mat, file)

        return img, mat, img_name
    
    def _save_files(self, img_scaled:np.ndarray, patch_label_arr_dots:np.ndarray, img_name:str):

        # Generate the detection ground truth dot map
        patch_label_arr_dots_all = patch_label_arr_dots[:, :, :].sum(axis=-1)
        # Save Dot maps

        patch_label_arr_dots.astype(np.uint8).dump(
            os.path.join(self.out_gt_dir, img_name + '_gt_dots.npy'))
        patch_label_arr_dots_all.astype(np.uint8).dump(
            os.path.join(self.out_gt_dir, img_name + '_gt_dots_all.npy'))
        
        if self.save_vis:
            self._save_visualization(img_scaled, patch_label_arr_dots, img_name)

        # Generate the Gaussian maps.
        # It is important to not do this for each class separately because this may result in intersections in the detection map
        _ = self._generate_gaussian_maps(img_scaled, patch_label_arr_dots, img_name)


    def _create_gt_files(self, file_dict:dict):

        img, mat, img_name = self._load_from_file_dict(file_dict)
        img_scaled, patch_label_arr_dots = self._create_grouped_gt(img, mat)     

        self._save_files(img_scaled, patch_label_arr_dots, img_name)        

    
    def _create_gt_files_from_crops(self, file_dict:dict):

        def _pad_im(x:np.ndarray):
            diff_h = self.win_size[0] - self.step_size[0]
            padt = diff_h // 2
            padb = diff_h - padt

            diff_w = self.win_size[1] - self.step_size[1]
            padl = diff_w // 2
            padr = diff_w - padl

            pad_type = "reflect"
            return np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), pad_type)

        patches_dir = os.path.join(self.root_dir, file_dict['patches_dir'])
        img_name = file_dict['img_id']
        array_files = natsorted([os.path.basename(file) for file in glob.glob(os.path.join(patches_dir, '*.npy'))])
        coords = [ self._get_coords_from_filename(file) for file in array_files]
        assert len(array_files) == len(coords), f'Number of files {len(array_files)} does not match number of coordinates {len(coords)}'

        img_whole, mat, img_name = self._load_from_file_dict(file_dict)
        img_whole, patch_label_arr_dots_whole = self._create_grouped_gt(img_whole, mat)

        patch_label_arr_dots_custom_all_whole, patch_label_arr_dots_custom_whole = self._generate_gaussian_maps(img_whole, patch_label_arr_dots_whole, img_name, save=False)
        img_whole = _pad_im(img_whole)
        patch_label_arr_dots_custom_whole = _pad_im(patch_label_arr_dots_custom_whole)
        patch_label_arr_dots_whole = _pad_im(patch_label_arr_dots_whole)


        for file, coord in zip(array_files, coords):
            
            crop_array = np.load(os.path.join(patches_dir, file), allow_pickle=True)[...,:5] # channels 0-3: img, 3: inst_map, 4: class_map
            assert np.array_equal(crop_array[...,:3], img_whole[coord[0][0]:coord[0][1], coord[1][0]:coord[1][1]])
            # we need the name.npy and the name_gt_dots.npy.

            gt_dots = patch_label_arr_dots_whole[coord[0][0]:coord[0][1], coord[1][0]:coord[1][1]]
            gt_dmap = patch_label_arr_dots_custom_whole[coord[0][0]:coord[0][1], coord[1][0]:coord[1][1]]

            # Concatenate to have:
            # channel 0,1,2: img RGB, 
            # channel 3: inst_map, 
            # channel 4: class_map
            # channel 5,6,7: gt_dots  ->  other cells,  healthy epithelial, malignant epithelial 
            # channel 8,9,10: gt_dmap ->  other cells,  healthy epithelial, malignant epithelial 

            new_gt = np.concatenate((crop_array, gt_dots[...,1:], gt_dmap[...,1:]), axis=-1)
            new_gt.astype(np.uint8).dump(os.path.join(patches_dir, file))




   
    def create_gt_dataset(self, files:list, root_dir:str, out_dir:str):

        self._set_root_dir(root_dir)
        self._create_out_dirs(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm(total=len(files), bar_format=pbar_format, ascii=True, position=0)

        if self.save_as_crops:

            extractor_func = self._create_gt_files_from_crops 
        else:
            extractor_func = self._create_gt_files
        
        for file in files:
            extractor_func(file)
            pbarx.update()

        pbarx.close()



if __name__ == "__main__":

    
    args = get_gt_extract_args()
    logger = create_logger()

    slurm_job = 'slurm job array' if os.environ.get('SLURM_JOB_ID') else 'local machine'
    slurm_array_task_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    slurm_array_job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    logger.info(f'Script started. Running in {slurm_job}. Task ID: {slurm_array_job_id+1} out of {slurm_array_task_count}')

    file_dict_list = parse_json_file(args.json_info_file_path)
    file_dict_list = divide_list(file_dict_list, slurm_array_task_count)[slurm_array_job_id]

    img_ids = natsorted([file['img_id'] for file in file_dict_list])
    logger.info(f'img_ids {img_ids}')

    gt_creator = LabelHandler(dataset=args.dataset,
                              nr_types=args.nr_types,
                              n_grouped_class_channels=len(args.grouping_dict) + 1,
                              class_group_mapping_dict=args.grouping_dict, # define mapping of cell classes in input to output
                              color_set={1: (0, 162, 232), 2: (255, 0, 0), 3: (0, 255, 0)}, # lymph: blue,  tumor: red, stromal: green
                              img_scale=args.img_scale, # Define rescaling rate of images
                              remove_duplicates=False,  # if True will remove duplicate of cells annotated within 5 pixel distance
                              save_vis=False,
                              save_as_crops=args.save_as_crops,
                              window_size=args.window_size,
                              step_size=args.step_size,)
    
    gt_creator.create_gt_dataset(file_dict_list, args.root_dir, args.out_root_dir)
    logger.info(f'Script Finished!!')
    # patches_dirs = [os.path.join(args.root_dir, tile['patches_dir']) for tile in file_dict_list]  
    
    # All patch directories can be found @ files[i]['patches_dir']

