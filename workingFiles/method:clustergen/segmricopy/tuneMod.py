from monai.networks.nets import DynUNet
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.data import Dataset, DataLoader, list_data_collate
from glob import glob
from utils.transforms import get_train_transforms, get_val_transforms, get_eval_post_transforms
from utils.training import train_NormSeg, test_NormSeg, HLoss
from utils.model import Normalizer, Normalizer_Deep, Normalizer_Deeper
from utils.plotting import input_check
import sys
import os
import torch
import argparse
import os.path as osp
import time

sys.path.append('./')

def tta(args, data, norm_model, segmenter, name):
    # Data loader
    if args.ttaug == True:
        val_ds = Dataset(
            data=data, transform=get_train_transforms(args.SPATIAL_SIZE, args.LABELS)) # train transforms include augmentation
    else:
        val_ds = Dataset(
            data=data, transform=get_val_transforms(args.SPATIAL_SIZE, args.ROI_CENTER_LOCATION))
    val_ds_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=2, collate_fn=list_data_collate,)

    # Settings
    params = list(norm_model.parameters())
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    if args.method == 'tent':
        loss_function = HLoss()

    optimizer = torch.optim.Adam(params, 0.0001)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    device = torch.device("cuda")

    # Load pre-trained models' states
    if args.model == 'NormSeg':
        norm_model.load_state_dict(torch.load(
            os.path.join(args.modelsDir + 'normalization_model_' + args.condition + '.pt')))
        segmenter.load_state_dict(torch.load(
            os.path.join(args.modelsDir + 'segmenter_model_' + args.condition + '.pt')))
    else:
        print('Segmenter plus TuneMod')
        segmenter.load_state_dict(torch.load(
            os.path.join(args.modelsDir + 'segmenter_model_only_' + args.condition + '.pt')))

    save_root = args.modelsDir + 'tta/'+ args.datasetName + '/' 
    if not osp.exists(save_root):
        os.makedirs(save_root)

    # Test-time adaptation
    for epoch in range(1, int(args.nepochs) + 1):
        init = time.time()
        loss, norm_input, input_img = train_NormSeg(
            norm_model, segmenter, val_ds_loader, optimizer, loss_function, device)
        dice_score = test_NormSeg(norm_model, segmenter,
                            val_ds_loader, dice_metric, get_eval_post_transforms(args.LABELS), device)
        if epoch % 50 == 0 or epoch == 1:
            input_check(epoch, 'tta', name,
                    norm_input, input_img, args.QC)
            torch.save(norm_model.state_dict(), save_root + name + '_tta_' + str(epoch) + 'e'
                        '_normalization_model_' + args.condition + '.pt') #change
        end = time.time()
        final_time = (end - init) / 60
        print('Loss: %s. Dice score: %s. Time per epoch: %s minutes' %
                (str(loss), str(dice_score), str(final_time)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataDir", type=str)
    parser.add_argument("-m", "--modelsDir", default="./models/")
    parser.add_argument("-c", "--QC", default="./checks/")
    parser.add_argument("-sub", "--subID", default=None)
    parser.add_argument("--model", type=str)
    parser.add_argument("--datasetName", type=str)
    parser.add_argument("--condition", type=str)
    parser.add_argument("--nepochs", type=str)
    parser.add_argument("--labels", default='atlas')
    parser.add_argument("--norm_arc", default='original')
    parser.add_argument("--method", default='TTADAE')
    parser.add_argument("--ttaug", default=False)
    parser.add_argument('--single_instance', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", default='0')

    args = parser.parse_args()

    args.SPATIAL_SIZE = [64, 128, 128]
    args.LABELS = {
        "vertical": 1,
        "superior longitudinal": 2,
        "inferior longitudinal": 3,
        "genioglossus": 4,
        "hyoglossus": 5,
        "background": 0,
    }
    if args.datasetName=='BeLong_T2w':
        args.ROI_CENTER_LOCATION = (100, 295, 210)
    elif args.datasetName[0:10]=='BeLong_T1w':
        args.ROI_CENTER_LOCATION = (100, 295, 210)
    elif args.datasetName[:4]=='EATT':
        args.ROI_CENTER_LOCATION = (100, 250, 90) 
    elif args.datasetName[:6]== 'Sydney':
        args.ROI_CENTER_LOCATION = (100, 250, 90)

    args.CHANNELS = 1

    # Data directory
    dataDir_img = args.dataDir + 'images/'
    if args.labels=='atlas':
        dataDir_seg = args.dataDir + 'labels/'
    elif args.labels=='suboptimal':
        dataDir_seg = args.dataDir + 'labels_suboptimal/'

    if args.subID == None:
        images = sorted(glob(os.path.join(dataDir_img, '*.nii.gz')))
        segs = sorted(glob(os.path.join(dataDir_seg, '*.nii.gz')))
    else:
        images = sorted(glob(os.path.join(dataDir_img, str(args.subID) + '*')))
        segs = sorted(glob(os.path.join(dataDir_seg, '*.nii.gz'))) # use only atlas labels
    
    if args.labels=='atlas':
        segs = segs * len(images)

    # Data prep
    val_files = [
        {'image': img, "label": seg} for img, seg in zip(images[:], segs[:])
    ]

    # Models
    if args.norm_arc == 'original':
        norm_model = Normalizer()
    elif args.norm_arc == 'deep':
        print('Deep normalization model')
        norm_model = Normalizer_Deep()
    elif args.norm_arc == 'deeper':
        norm_model = Normalizer_Deeper()
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
    # print(args.single_instance)
    if args.single_instance == True:
        for i in range(len(val_files)):
            name = os.path.split(val_files[i]['image'])[1][:-7]
            if args.norm_arc == 'deep':
                name = name + '_deep'
            if args.norm_arc == 'deeper':
                name = name + '_deeper'
            ind_data = [val_files[i]]
            tta(args, ind_data, norm_model, segmenter, name)
    else:
        n_samples = len(val_files)
        name = str(n_samples) + 'samples_seed' + str(args.seed)
        
        if args.norm_arc == 'deep':
                name = name + '_deep'
        if args.norm_arc == 'deeper':
            name = name + '_deeper'
        print('Number of samples is %s' % len(val_files))
        tta(args, val_files, norm_model, segmenter, name)
        
if __name__ == '__main__':
    main()
