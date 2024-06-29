"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
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


def get_data_path(root='examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    # lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    transform_params, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm, transform_params

def main(rank, opt, name='examples'):
    import os.path as osp

    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path = get_data_path(name)
    im_path, lm_path = im_path, lm_path
    lm3d_std = load_lm3d(opt.bfm_folder) 

    save_dir = osp.join(osp.dirname(name), "deep3dface")
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(len(im_path)):
        print(i, im_path[i], lm_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        if not os.path.isfile(lm_path[i]):
            continue
        im_tensor, lm_tensor, transform_params = read_data(im_path[i], lm_path[i], lm3d_std)
        data = {
            'imgs': im_tensor,
            'lms': lm_tensor,
            'transform_params': transform_params
        }
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        # visuals = model.get_current_visuals()  # get image results
        # visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1], 
        #     save_results=True, count=i, name=img_name, add_image=False)

        # model.save_mesh(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
        # model.save_coeff(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0), img_name+'.mat')) # save predicted coefficients

        ## -----For our customized results saving-------------------
        model.save_coeff(os.path.join(save_dir, img_name + '.mat')) # save predicted coefficients
        
        

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(0, opt, opt.img_folder)
    
