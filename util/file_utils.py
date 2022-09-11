'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-11 09:45:30
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


import os
import os.path as osp
from scipy.io import loadmat, savemat
import numpy as np
import torch
from PIL import Image
from .preprocess import align_img


def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W, H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    
    transform_params, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    
    trans_params = np.array([float(item) for item in np.hsplit(transform_params, 5)])
    return im, lm, trans_params


def get_folder_list(root_dir, 
                    folders_name_file=None, 
                    subfolder_name="",
                    suffix=None):
    if folders_name_file is not None:
        folder_lines = open(folders_name_file).read().splitlines()
        folder_list = [osp.join(root_dir, folder_name, subfolder_name) \
                       for folder_name in folder_lines]
    else:
        folder_list = [osp.join(entry.path, subfolder_name) \
                       for entry in os.scandir(root_dir) if entry.is_dir()]
        folder_list = sorted(folder_list)
    
    if suffix is not None:
        folder_list = [f"{folder.rstrip('/')}{suffix}" for folder in folder_list]
    return folder_list


def get_splited_filelists(filelists: list, 
                          split_length=100, 
                          disable_split=False):
    if disable_split: # return the original list
        return [filelists]
    
    data_splited = [filelists[i:i + split_length] for i in range(0, len(filelists), split_length)]
    return data_splited


def fetch_subsplit_filelists(root_dir, split_length, **kwargs):
    """Get the subsplit filelists

    Args:
        root_dir (str): root directory
        folders_name_file (str, optional): folder name file. Defaults to None.
        subfolder_name (str, optional): subfolder name. Defaults to "".

    Returns:
        list: list contains all filelists
    """
    folder_list = get_folder_list(root_dir, **kwargs)

    splited_list = get_splited_filelists(folder_list, split_length, **kwargs)
    return splited_list