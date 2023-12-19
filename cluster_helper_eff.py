import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm as tqdm
import skimage.io as io
from skimage.measure import label
from sklearn.cluster import KMeans
from utils import create_logger, empty_trash


import numba as nb

# short-circuiting replacement for np.any()
@nb.jit(nopython=True)
def sc_any(array):
    for x in array.flat:
        if x:
            return True
    return False

logger = create_logger('cluster_helper')

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collect_features_by_class(model, simple_train_loader, feature_indx_list, n_classes):
    img_has_centroids = [[False]*len(simple_train_loader.dataset) for i in range(n_classes)] 
    features_list = [[None]*len(simple_train_loader.dataset) for i in range(n_classes)]    # imp to avoid referencing the same list in all entries
    coord_list = [[0]*len(simple_train_loader.dataset) for i in range(n_classes)] # imp to avoid referencing the same list in all entries
    model.eval()
    idx = 0
    with torch.no_grad():
            for (img,__,gt_dots,__, padding) in simple_train_loader:
                # padding: padding added to the image to make sure it is a multiple of 16 (corresponding to 4 max pool layers)
                pad_y1  = padding[0].numpy()[0]
                pad_y2  = padding[1].numpy()[0]
                pad_x1  = padding[2].numpy()[0]
                pad_x2  = padding[3].numpy()[0]

                # get the ground truth dot map for all cells without the padding
                gt_dots = gt_dots[:,:,pad_y1:gt_dots.shape[-2]-pad_y2,pad_x1:gt_dots.shape[-1]-pad_x2]
                gt_dots = gt_dots.cpu().numpy()
                # get the image features from the model
                et_dmap_lst, img_feats = model(x=img, feat_indx_list=feature_indx_list)
                img_feats = img_feats[:,:,2:-2,2:-2]
                img_feats = img_feats[:,:,pad_y1:img_feats.shape[-2]-pad_y2,pad_x1:img_feats.shape[-1]-pad_x2]
                img_feats = img_feats.cpu()

                del et_dmap_lst 

                for j in range(len(img_feats)):
                    img_feat = img_feats[j].permute((1,2,0)).numpy()
                    gt_dot = gt_dots[j]

                    has_centroids = [sc_any(gt_dot[i,...]) for i in range(gt_dot.shape[0])]
                    if any(has_centroids):
                        points_all = np.where(gt_dot > 0)
                        for s in range(len(has_centroids)):
                            if not has_centroids[s]:
                                coord_list[s][idx] = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
                                continue
                            pos = (points_all[0] == s)
                            points = points_all[1][pos], points_all[2][pos]
                            img_has_centroids[s][idx] = True
                            coord_list[s][idx] = points
                            features_list[s][idx] = img_feat[points]
                    else:
                        for s in range(len(has_centroids)):
                            coord_list[s][idx] = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

                    idx += 1

    empty_trash()
    features_list = [np.vstack(list(filter(lambda ele:ele is not None, class_features))) for class_features in features_list]
                                
    logger.info('----------------Finished extracting features')
    return features_list, img_has_centroids, coord_list
    
def cluster(features_array, img_has_centroids,coord_list, n_clusters, prev_centroids):
    # For each class, get all features and do kmeans clustering, then use the fitted kmeans to get the pseudo clustering label for each cell
    cluster_centers_all = None
    pseudo_labels_list = [[None]*len(img_has_centroids[0]) for i in range(len(img_has_centroids))]
    for s in range(len(features_array)):

        # To have a more stable clustering, we initialize kmeans centroids with previous clustering centroids
        if(prev_centroids is None):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            
        else:
            kmeans = KMeans(n_clusters=n_clusters, init=prev_centroids[s*n_clusters:s*n_clusters+n_clusters])
        preds = kmeans.fit_predict(np.nan_to_num(features_array[s], copy=True, nan=0.0)) if features_array[s] is not None else None 

        # Predict the cluster label for each cell
        pos = 0
        for i in range(len(img_has_centroids[s])):
            if not img_has_centroids[s][i]:
                continue
            n_centroids = len(coord_list[s][i][0])

            pseudo_labels_list[s][i] = preds[pos:pos+n_centroids]
            pos += n_centroids

        if(cluster_centers_all is None):
            cluster_centers_all = kmeans.cluster_centers_
        else:
            cluster_centers_all = np.concatenate((cluster_centers_all, kmeans.cluster_centers_), axis=0)
    logger.info('----------------Finished K-means clustering')
    # return the cluster labels and the centroids
    return pseudo_labels_list, cluster_centers_all

def create_pseudo_lbl_gt(simple_train_loader, pseudo_labels_list, coord_list, n_clusters, out_dir):
    # logger.info('----------------Start saving pseudo-label files')
    n_subclasses = len(pseudo_labels_list) * n_clusters # number of sub classes is number of cell classes * number of clusters
    i = 0
    simple_train_loader = DataLoader(simple_train_loader.dataset, batch_size=30, shuffle=False, num_workers=14)
    with tqdm(simple_train_loader, unit='batch', desc=f"Clustering save", ascii=' =', dynamic_ncols=True) as batch_loader:
        for __,gt_dmaps,__, img_names, paddings in batch_loader:
            ''' 
                img: input image
                gt_dmap: ground truth map for cell classes (lymphocytes, epithelial/tumor, stromal) with dilated dots. This can be a binary mask or a density map ( in which case it will be converted to a binary mask)
                gt_dots: ground truth binary dot map for cell classes (lymphocytes, epithelial/tumor, stromal). 
                img_name: img filename
                padding: padding added to the image to make sure it is a multiple of 16 (corresponding to 4 max pool layers)
            '''
            pads_y1  = paddings[0].numpy()
            pads_y2  = paddings[1].numpy()
            pads_x1  = paddings[2].numpy()
            pads_x2  = paddings[3].numpy()
            # get the ground truth maps without the padding
            # Convert ground truth maps to binary mask (in case they were density maps)
            gt_dmaps = gt_dmaps > 0

            for j in range(len(img_names)):
                gt_dmap = gt_dmaps[j]
                pad_y1  = pads_y1[j]
                pad_y2  = pads_y2[j]
                pad_x1  = pads_x1[j]
                pad_x2  = pads_x2[j]
                img_name = img_names[j]

                gt_dmap = gt_dmap[...,pad_y1:gt_dmap.shape[-2]-pad_y2,pad_x1:gt_dmap.shape[-1]-pad_x2]

                # Initialize the ground truth maps for the clustering sub-classes
                gt_dmap_all =  gt_dmap.max(0)[0] # dim 0 is squeezed in loop
                gt_dmap_all = gt_dmap_all.numpy().squeeze()
                gt_dmap_subclasses = np.zeros((gt_dmap_all.shape[0], gt_dmap_all.shape[1], n_subclasses))

                label_comp = label(gt_dmap_all)
                cci = 0
                for s in range(len(pseudo_labels_list)):
                    pseudo_labels = pseudo_labels_list[s][i]
                    if(pseudo_labels is None):
                        cci += n_clusters
                        continue
                    points = coord_list[s][i]
                    for c in range(n_clusters):

                        # Set the dilated dot map (mask) for the cell-sub-cluster.
                        gt_map_tmp = np.zeros((gt_dmap_subclasses.shape[0],gt_dmap_subclasses.shape[1]))
                        # Assign to each connected component the same label as the ground truth dot in that cell
                        comp_in_cluster = label_comp[(points[0][(pseudo_labels == c)], points[1][(pseudo_labels == c)])]
                        for comp in comp_in_cluster:
                            gt_map_tmp[label_comp==comp] = 1
                        gt_dmap_subclasses[:,:,cci] = gt_map_tmp
                        cci += 1
                        # Save map as image. Useful for debugging.
                        # io.imsave(os.path.join(out_dir, img_name_list[i].replace('.png','_gt_dmap_s'+str(s)+'_c'+str(c)+'.png')), (gt_map_tmp*255).astype(np.uint8))

                # Save generated ground truth maps for the current image
                file_path = os.path.sep.join(os.path.normpath(img_name).split(os.path.sep)[-2:])
                file_path = os.path.join(out_dir, file_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                np.save(file_path, gt_dmap_subclasses.astype(np.uint8))
                i += 1
    logger.info('----------------Finished saving pseudo-label files')
        

def perform_clustering(model, simple_train_loader, n_clusters, n_classes, feature_indx_list, out_dir, prev_centroids, using_crops:bool=False):
    '''
        model: MCSpatNet model being trained
        simple_train_loader: data loader for training data to iterate over input
        n_clusters: number of clusters per class
        n_classes: number of cell classes
        feature_indx_list: features to use in clustering [ feature_code = {'decoder':0, 'cell-detect':1, 'class':2, 'subclass':3, 'k-cell':4} ]
        out_dir: directory path to output generated pseudo ground truth
        prev_centroids: previous epoch clustering centroids
    '''

    # Get the features to use for clustering

    features, img_has_centroids, coord_list = collect_features_by_class(model, simple_train_loader, feature_indx_list, n_classes)

    # Do the clustering: get the centroids for the new clusters and the pseudo ground truth labels
    pseudo_labels_list, centroids = cluster(features, img_has_centroids,coord_list, n_clusters, prev_centroids)

    # Save the pseudo ground truth labels to the file system to be able to use in training
    create_pseudo_lbl_gt(simple_train_loader, pseudo_labels_list, coord_list, n_clusters, out_dir)

    return centroids

