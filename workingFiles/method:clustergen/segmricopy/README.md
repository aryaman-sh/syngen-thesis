# Robust and data-efficient training of deep learning-based medical image segmentation models

This repository contains code for training segmentation models using the MONAI framework.

## Installation and Requirements

Before model training, please set up a conda environment as below:

- Clone this repository to your local machine:
```bash
git clone git@github.com:felenitaribeiro/SegMRI.git
```

- Install miniconda:
```bash
cd vessel_code
bash miniconda-setup.sh
```

- Then create your conda environment:

```bash
conda env create -f environment.yml
conda activate robust_dl
```

## Model training

To train a model, run the following command:

```bash
python train.py -d dataset/ --model segmenter --data cropped --nepochs 2 --datatype tongue
```

The arguments are as follows:

- `-d`: path to the dataset folder
- `--model`: model to be trained. Options are `segmenter` and `NormSeg`
- `--data`: name of the data to be used. 
- `--nepochs`: number of epochs to train the model
- `--datatype`: type of data to be used. Options are `tongue` and `prostate` # soon to be added `abdomen`

## Inference

To run inference using a pre-trained model, run the following command:

```bash
python inference.py -d dataset/ -p predictions/ --model segmenter --data cropped --datasetName BeLong --nepochs 2 --datatype tongue
```

The arguments are as follows:

- `-d`: path to the dataset folder
- `-p`: path to the predictions folder
- `--model`: trained model. Options are `segmenter` and `NormSeg`
- `--data`: name of the data to be used.
- `--datasetName`: name of the dataset to be used. Options are `BeLong` and `BeLong`
- `--nepochs`: number of epochs the model was trained
- `--datatype`: type of data to be used. Options are `tongue` and `prostate` # soon to be added `abdomen`

