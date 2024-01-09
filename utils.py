import json
import os
import argparse
import logging
import sys
from natsort import natsorted
import glob
import gc
import torch
import numpy as np
import random

def get_npy_files(root_dir, file_type='npy'):
    # Use glob to find all .npy files in root_dir and its subdirectories
    npy_files = glob.glob(os.path.join(root_dir, '**', f'*.{file_type}'), recursive=True)

    return natsorted(npy_files)

def get_kmaps_extract_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/storage/homefs/jg23p152/project",)
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset folder")
    parser.add_argument("--out_root_dir", type=str, default="k_func_maps",
                        help="name of the output folder")
    # parser.add_argument("--dataset", type=str, default="lizard",
    #                     help="Name of the dataset")
    parser.add_argument("--nr_types", type=int, default=3,
                        help="Number of cell types, not counting background as one type")
    # parser.add_argument("--grouping_dict", type=str, default="1:1,2:2,3:3")

    args = parser.parse_args()
    # args.grouping_dict = {int(k): [int(i) for i in v.split('-')] for k, v in [i.split(':') for i in args.grouping_dict.split(',')]}
    return args

def get_gt_extract_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/storage/homefs/jg23p152/project",)
    parser.add_argument("--json_info_file_path", type=str, default="/storage/homefs/jg23p152/project/Data/Split_files/training_set_files_debug.json",
                        help="Path to the json with the dataset info")
    parser.add_argument("--out_root_dir", type=str, default="/MCSpatNet/datasets/Lizard",
                        help="Path to the output root directory")
    parser.add_argument("--dataset", type=str, default="lizard",
                        help="Name of the dataset")
    parser.add_argument("--nr_types", type=int, default=4,
                        help="Number of cell types, counting background as one type")
    parser.add_argument("--grouping_dict", type=str, default="1:1,2:2,3:3")
    parser.add_argument("--img_scale", type=float, default=1.0)
    parser.add_argument("--save_as_crops", action='store_true', help='Set to save GT atcrop level instead of image level')
    parser.add_argument("--window_size", type=int, default=540, help='Size of the window to crop')
    parser.add_argument("--step_size", type=int, default=164, help='Step size for cropping')
    args = parser.parse_args()
    args.grouping_dict = {int(k): [int(i) for i in v.split('-')] for k, v in [i.split(':') for i in args.grouping_dict.split(',')]}
    return args

def get_train_args():
    parser= argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/storage/homefs/jg23p152/project', help='The root directory.')
    parser.add_argument('--checkpoints_root_dir', type=str, default='/storage/homefs/jg23p152/project/MCSpatNet/MCSpatNet_checkpoints', help='The root directory for all training output.')
    parser.add_argument('--checkpoints_folder_name', type=str, default='mcspatnet_consep_1', help='The name of the folder that will be created under <checkpoints_root_dir> to hold output from current training instance.')
    parser.add_argument('--model_param_path', type=str, default=None, help='path of a previous checkpoint to continue training')
    parser.add_argument('--clustering_pseudo_gt_root', type=str, default='/storage/homefs/jg23p152/project/MCSpatNet/MCSpatNet_epoch_subclasses', help='path to save clustering pseudo ground truth')
    parser.add_argument('--train_data_root', type=str, default='/storage/homefs/jg23p152/project/MCSpatNet/datasets/Lizard', help='path to training data')
    parser.add_argument('--test_data_root', type=str, default='/storage/homefs/jg23p152/project/MCSpatNet/datasets/Lizard', help='path to testing data')
    parser.add_argument('--train_split_filepath', type=str, default='/storage/homefs/jg23p152/project/Data/Split_files/cv0_train_files.json', help='path to training data split file')
    parser.add_argument('--test_split_filepath', type=str, default='/storage/homefs/jg23p152/project/Data/Split_files/cv0_val_files.json', help='path to testing data split file')
    parser.add_argument('--cluster_data_filepath', type=str, default=None, help='path to data to be clustered for pseudo-labeling file')

    parser.add_argument("--nr_types", type=int, default=2, help="Number of cell types, not counting background as one type")
    parser.add_argument('--cropped_data', action='store_true', help='Set if using with pre-cropped data')
    parser.add_argument('--crop_size', type=int, default=442, help='Images will be cropped to this size for training')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs. Use 300 for CoNSeP dataset.')
    parser.add_argument('--start_finetuning', type=int, default=1, help='Epoch index when to start finetuning the network')
    parser.add_argument('--batch_size', type=str, default='1,1', help='batch sizes with frozen encoder and finetuning separated by comma')
    parser.add_argument('--lr', type=str, default='5e-5,1e-5', help='learning rates with frozen encoder and finetuning separated by comma')
    parser.add_argument('--cell_code', type=str, default='0:others,1:epit_healthy,2:epit_maligant', help='dictionary of cell classes')
    parser.add_argument('--lamda_class', type=float, default=1.0, help='weight of dice loss (classification)')
    parser.add_argument('--lamda_detect', type=float, default=1.0, help='weight of dice loss (detection)')
    parser.add_argument('--lamda_subclass', type=float, default=1.0, help='weight of dice loss (deep clustering pseudo-labeling)')
    parser.add_argument('--lamda_kfunc', type=float, default=1.0, help='weight of K-function L1-loss')
    parser.add_argument('--use_dice_loss', action='store_true', help='Set to use dice loss for classification and detection')
    parser.add_argument('--use_ce_loss', action='store_true', help='Set to use cross entropy loss for classification and detection')

    parser.add_argument('--seed', type=int, default=123, help='random seed')

    args = parser.parse_args()
    args.cell_code = {int(s.split(':')[0]):s.split(':')[1] for s in args.cell_code.split(',')}
    args.batch_size = [int(s) for s in args.batch_size.split(',')]
    args.lr = [float(s) for s in args.lr.split(',')]
    return args

def create_logger(name:str=__name__):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)s: %(message)s',
                                    "%b-%d %H:%M:%S")
        # Write logs to the SLURM output file
        file_handler = logging.StreamHandler(sys.stdout)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def divide_list(lst, n):
    division_size = len(lst) // n
    divisions = [lst[i * division_size:(i + 1) * division_size] for i in range(n - 1)]
    divisions.append(lst[(n - 1) * division_size:])
    return divisions


def parse_json_file(file_path):
    """
    Parses the file specified in path, with some additional security checks.
    :param file_path: File path to parse
    :returns : The contents in JSON format, or None if the file is empty.
    """
    if not file_path or os.stat(file_path).st_size == 0:
        return None

    with open(file_path, "r") as json_file:
        file_contents = json_file.read()

    return parse_json(file_contents)


def _replace_single_quotes(text):
    replaced_text = text.replace("'", '"')

    return replaced_text

def parse_json(text):
    """
    Parses the specified text as JSON, with some additional security checks.
    :param text: Text to parse.
    :returns : The parsed results, or None if the string is empty.
    """
    if not text:
        return None
    else:
        try:
            return json.loads(text)
        except Exception:
            return json.loads(_replace_single_quotes(text))

def save_json(path:str, file_name:str, object):
    os.makedirs(os.path.join(path), exist_ok=True)
    with open(os.path.join(path, file_name), 'w') as f:
        json.dump(object, f)

def empty_trash():
    gc.collect()
    torch.cuda.empty_cache()

def seed_everything(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)