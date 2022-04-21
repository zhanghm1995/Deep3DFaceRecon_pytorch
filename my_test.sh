set -x

img_folder="./HDTF_preprocessed/WDA_MartinHeinrich_000/face_image"

python my_test.py --name face_recon_feat0.2_augment --epoch=20 --gpu_ids 0 \
                  --img_folder=$img_folder