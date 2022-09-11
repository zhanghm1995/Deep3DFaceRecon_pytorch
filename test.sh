set -x

# img_folder="./datasets/examples"

# python test.py --name face_recon_feat0.2_augment --epoch=20 --gpu_ids 0 \
#                   --img_folder=$img_folder

img_folder="/home/zhanghm/Temp/cv-fighter/Cheat-Sheet-For-FFmpeg/vox1/train/id10198#Y8MjZMcg5J8#005118#005453"
python test.py --name HDTFDataset_default --epoch=20 --gpu_ids 0 \
               --img_folder=$img_folder