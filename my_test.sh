
# img_folder="./examples/face_image"

# python my_test.py --name face_recon_feat0.2_augment --epoch=20 --gpu_ids 0 \
#                   --img_folder=$img_folder

img_folder="/home/zhanghm/Temp/cv-fighter/Cheat-Sheet-For-FFmpeg/vox1/train/id10198#Y8MjZMcg5J8#005118#005453"

set -x
python my_test.py --name face_recon_feat0.2_augment --epoch=20 --gpu_ids 0 \
                  --img_folder=$img_folder