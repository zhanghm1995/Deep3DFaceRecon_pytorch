"""This script is the data preparation script for Deep3DFaceRecon_pytorch
"""

import os 
import numpy as np
import argparse
from util.detect_lm68 import detect_68p,load_lm_graph
from util.skin_mask import get_skin_mask
from util.generate_list import check_list, write_list
import warnings
warnings.filterwarnings("ignore") 

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='datasets', help='root directory for training data')
parser.add_argument('--img_folder', nargs="+", required=True, help='folders of training images')
parser.add_argument('--mode', type=str, default='train', help='train or val')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def data_prepare(folder_list,mode):

    lm_sess,input_op,output_op = load_lm_graph('./checkpoints/lm_model/68lm_detector.pb') # load a tensorflow version 68-landmark detector

    for img_folder in folder_list:
        detect_68p(img_folder,lm_sess,input_op,output_op) # detect landmarks for images
        get_skin_mask(img_folder) # generate skin attention mask for images

    # create files that record path to all training data
    msks_list = []
    for img_folder in folder_list:
        path = os.path.join(img_folder, 'mask')
        msks_list += ['/'.join([img_folder, 'mask', i]) for i in sorted(os.listdir(path)) if 'jpg' in i or 
                                                    'png' in i or 'jpeg' in i or 'PNG' in i]

    imgs_list = [i.replace('mask/', '') for i in msks_list]
    lms_list = [i.replace('mask', 'landmarks') for i in msks_list]
    lms_list = ['.'.join(i.split('.')[:-1]) + '.txt' for i in lms_list]
    
    lms_list_final, imgs_list_final, msks_list_final = check_list(lms_list, imgs_list, msks_list) # check if the path is valid
    write_list(lms_list_final, imgs_list_final, msks_list_final, mode=mode) # save files


def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


def generate_five_landmarks():
    """
    generate five landmarks for each image
    """
    import dlib
    from skimage import io

    dlib_landmark_model = './checkpoints/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(dlib_landmark_model)
    detector = dlib.get_frontal_face_detector()

    img_path = "/home/zhanghm/Research/VideoMAE/VideoMAE/data/HDTF_preprocessed/RD_Radio1_000/face_image/remove/000000.jpg"
    img = io.imread(img_path)
    dets = detector(img, 0)

    if list(dets) == []:
        print("No face detected")
    pts = predictor(img, dets[0]).parts()
    pts = np.array([[pt.x, pt.y] for pt in pts]) # (68, 2) facial landmarks

    five_lms = extract_5p(pts) # (5, 2) five landmarks

    ## save the results
    np.savetxt("temp.txt", five_lms, fmt="%.2f", delimiter=' ')


if __name__ == '__main__':
    # generate_five_landmarks()
    print('Datasets:',opt.img_folder)
    data_prepare([os.path.join(opt.data_root,folder) for folder in opt.img_folder],opt.mode)
