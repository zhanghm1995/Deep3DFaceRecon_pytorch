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