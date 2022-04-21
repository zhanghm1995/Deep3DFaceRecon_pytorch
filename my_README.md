## Deep3DFace

## Dependices
- Install insightface
```
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/
```

## Get Started
1. Download the pretrained model, and place it into `./checkpoints` folder.
2. Download the BFM files.
```bash
cd ./BFM
ln -s <your_BFM>/BFM/01_MorphableModel.mat ./
ln -s <your_BFM>/BFM/BFM_model_front.mat ./
```
3. Run the process script
Modify the `--img_folder` to your image folder in `my_test.sh` script, and then run the command below.
```bash
bash my_test.sh
```
Note: for each face image, you need create a `.txt` file with the same name with the image, and put the landmark points coordinate in the file.