'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-04-21 19:50:50
Email: haimingzhang@link.cuhk.edu.cn
Description: My customized test script
'''


"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
import os.path as osp
from tqdm import tqdm
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat
from util.file_utils import read_data


def get_data_path(root='examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    # lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path


def main(rank, opt, name='examples'):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path = get_data_path(name)
    lm3d_std = load_lm3d(opt.bfm_folder) 

    save_dir = osp.join(osp.dirname(name), "deep3dface")
    os.makedirs(save_dir, exist_ok=True)

    prog_bar = tqdm(range(len(im_path)), total=len(im_path))
    for i in prog_bar:
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        if not os.path.isfile(lm_path[i]):
            continue
        
        prog_bar.set_description(f"{i} {im_path[i]}")

        im_tensor, lm_tensor, transform_params = read_data(im_path[i], lm_path[i], lm3d_std)
        data = {
            'imgs': im_tensor,
            'lms': lm_tensor,
            'trans_params': transform_params
        }
        
        ## Forward
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        # visuals = model.get_current_visuals()  # get image results
        # visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1], 
        #     save_results=True, count=i, name=img_name, add_image=False)

        pred_coeff = {key:model.pred_coeffs_dict[key].cpu().numpy() for key in model.pred_coeffs_dict}
        pred_coeff = np.concatenate([
                pred_coeff['id'], 
                pred_coeff['exp'], 
                pred_coeff['tex'], 
                pred_coeff['angle'],
                pred_coeff['gamma'],
                pred_coeff['trans']], 1)
        
        trans_params = transform_params[None]
        save_path = osp.join(save_dir, f"{img_name}.mat")
        savemat(save_path,
                {'coeff':pred_coeff, 'transform_params':trans_params})
        
        # model.save_coeff(os.path.join(save_dir, img_name + '.mat'))


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(0, opt, opt.img_folder)
    