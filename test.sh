set -x

img_folder="./datasets/examples"

python test.py --name face_recon_feat0.2_augment --epoch=20 --gpu_ids 0 \
                  --img_folder=$img_folder