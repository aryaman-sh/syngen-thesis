from monai.networks.nets import DynUNet
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.data import Dataset, DataLoader, list_data_collate
from glob import glob
from utils.transforms import get_train_transforms, get_val_transforms, get_eval_post_transforms, labels_to_int
from utils.training import train_NormSeg, train_segmenter, test_NormSeg, test_segmenter
from utils.model import Normalizer
from utils.plotting import input_check
import sys
import os
import torch
import argparse
import time

sys.path.append('./')


def dataloader(path, SPATIAL_SIZE, LABELS, ROI_CENTER_LOCATION, DATASETNAME):
    ''' Dataloader for training
    path: path to the data directory
    SPATIAL_SIZE: final size of the input image
    LABELS: labels
    ROI_CENTER_LOCATION: center of the ROI
    '''
    # Data directory
    dataDir_img = path + 'images/'
    dataDir_seg = path + 'labels/'

    images = sorted(glob(os.path.join(dataDir_img, '*.nii*')))
    segs = sorted(glob(os.path.join(dataDir_seg, '*.nii*')))

    for i in range(len(images)):
        assert images[i].split('/')[-1] == segs[i].split('/')[-1]
    print('All file names are correct')
    
    for i in range(len(segs)):
        labels_to_int(segs[i])

    samples = len(images)

    if DATASETNAME == 'organ':
        # Data prep
        print("Dataset correct loop")
        train_files = [
            {'image': img, "label": seg} for img, seg in zip(images[:int(samples*.50)], segs[:int(samples*.50)])
        ]
        val_files = [
            {'image': img, "label": seg} for img, seg in zip(images[int(samples*.50):], segs[int(samples*.50):]) 
        ]
    else:
        # Train Files == Val Files
        train_files = [
            {'image': img, "label": seg} for img, seg in zip(images[:], segs[:])
        ]
        val_files = [
            {'image': img, "label": seg} for img, seg in zip(images[:], segs[:]) 
        ]

    n_samples = len(train_files)
    # Dataloader
    train_ds = Dataset(
        data=train_files, transform=get_train_transforms(SPATIAL_SIZE, LABELS))
    train_ds_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                                 num_workers=2, collate_fn=list_data_collate,
                                 pin_memory=torch.cuda.is_available())

    val_ds = Dataset(
        data=val_files, transform=get_val_transforms(SPATIAL_SIZE, ROI_CENTER_LOCATION))
    val_ds_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                               num_workers=2, collate_fn=list_data_collate,)

    # including real data to evaluate those metrics
    real_val_img_dir = '/scratch/itee/uqasha24/synthetic-generalisation/fatdata_split/eval/images/'
    real_val_lab_dir = '/scratch/itee/uqasha24/synthetic-generalisation/fatdata_split/eval/labels/'
    images_real = sorted(glob(os.path.join(real_val_img_dir, '*.nii*')))
    segs_real = sorted(glob(os.path.join(real_val_lab_dir, '*.nii*')))
    val_files_real = [
            {'image': img, "label": seg} for img, seg in zip(images_real[:], segs_real[:])
        ]
    val_ds_real = Dataset(
        data=val_files_real, transform=get_val_transforms(SPATIAL_SIZE, ROI_CENTER_LOCATION))
    val_ds_loader_real = DataLoader(val_ds_real, batch_size=1, shuffle=False,
                               num_workers=2, collate_fn=list_data_collate,)
    return train_ds_loader, val_ds_loader, n_samples, val_ds_loader_real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataDir", type=str)
    parser.add_argument("-p", "--modelsDir", default="./models/")
    parser.add_argument("-c", "--QC", default="./checks/")
    parser.add_argument("--model", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--nepochs", type=str)
    parser.add_argument("--datatype", type=str, default='tongue')
    parser.add_argument("--datasetName", type=str)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    args.CHANNELS = 1

    if args.datatype == 'tongue':
        args.SPATIAL_SIZE = [64, 192, 192]
        args.LABELS = {
            "vertical": 1,
            "superior longitudinal": 2,
            "inferior longitudinal": 3,
            "genioglossus": 4,
            "hyoglossus": 5,
            "background": 0,
        }
        args.roi_center = (96, 135, 220)
        
    if args.datatype == 'prostate':
        args.SPATIAL_SIZE = [192, 192, 64]
        args.LABELS = {
            "prostate": 1,
            "background": 0,
        }
        args.roi_center = (220, 250, 32)

    if args.datatype == 'organ':
        args.SPATIAL_SIZE = [192, 192, 64] #Should be multiples of 32
        args.LABELS = {
            "organ1": 1, "organ2": 2, "organ3":3,
            "background": 0,
        }
        # Using the center as roi Dosent matter for training so ehhh
        args.roi_center = (144, 117, 61)



    # Models
    norm_model = Normalizer()
    segmenter = DynUNet(
        spatial_dims=3,
        in_channels=args.CHANNELS,
        out_channels=len(args.LABELS),
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
        dropout=.2,
    )
    if args.model == 'NormSeg':
        params = list(norm_model.parameters()) + list(segmenter.parameters())
    else:
        params = list(segmenter.parameters())

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(params, 0.001) # changed LR
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda') # enforce cuda
    train_loader, val_loader, n_samples, val_loader_real = dataloader(
            args.dataDir, args.SPATIAL_SIZE, args.LABELS, args.roi_center, args.datasetName)
    
    print('Number of samples is %s' % n_samples)

    for epoch in range(1, int(args.nepochs) + 1):
        init = time.time()
        if args.model == 'NormSeg':
            loss, norm_input, input_img = eval('train_' + args.model)(norm_model, segmenter, train_loader,
                                                                    optimizer, loss_function, device)
            dice_score = eval('test_' + args.model)(norm_model,
                                                segmenter, val_loader, dice_metric, get_eval_post_transforms(args.LABELS), device)

        else:
            loss, norm_input, input_img = eval('train_' + args.model)(segmenter, train_loader,
                                                                    optimizer, loss_function, device)
            dice_score = eval('test_' + args.model)(segmenter, val_loader, dice_metric, get_eval_post_transforms(args.LABELS), device)
            dice_score_real = eval('test_' + args.model)(segmenter, val_loader_real, dice_metric, get_eval_post_transforms(args.LABELS), device)                                              
        

        if epoch % 5 == 0 or epoch == 1:
            input_check(epoch, args.model, args.data,
                        norm_input, input_img, args.QC)
            if args.model == 'NormSeg':
                torch.save(norm_model.state_dict(), args.modelsDir +
                        'normalization_model_' + str(epoch) + 'e_' + args.data + '.pt')
                torch.save(segmenter.state_dict(), args.modelsDir +
                        'segmenter_model_' + str(epoch) + 'e_' + args.data + '.pt')
            else:
                if args.datasetName == 'BeLong_T2w':
                    torch.save(segmenter.state_dict(), args.modelsDir +
                            'segmenter_model_only_' + str(epoch) + 'e_' + args.data + '.pt')
                else:
                    torch.save(segmenter.state_dict(), args.modelsDir +
                            'train_samples_' + str(n_samples) + '_samples_seed' + str(args.seed) + '_segmenter_model_only_' + str(epoch) + 'e_' + str(args.data) + '_' + str(args.datasetName) + '.pt')
        end = time.time()
        final_time = (end - init) / 60
        print('Loss: %s. Dice score in the validation set (synthetic): %s. (real eval): %s. Time per epoch: %s minutes' %
              (str(loss), str(dice_score), str(dice_score_real),str(final_time)),)


if __name__ == '__main__':
    main()
