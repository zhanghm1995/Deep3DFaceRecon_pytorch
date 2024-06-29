set -x
img_folder=./data/data_preprocessed/May/face_image
python process_HDTF.py --name face_recon_feat0.2_augment --epoch=20 --gpu_ids 0 \
               --img_folder=$img_folder

