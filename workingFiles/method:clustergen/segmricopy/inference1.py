"""
Made changes to inference.py
"""

import argparse
import os
import numpy as np
import torch
import nibabel as nib
import torch.nn as nn
import os.path as osp
from utils.transforms import get_infer_transforms, get_post_transforms, labels_to_int
from utils.model import Normalizer, Normalizer_Deep, Normalizer_Deeper, Normalizer_Deeper2x, segmenter_model
from glob import glob
from monai.data import Dataset, DataLoader
from monai.networks.nets import DynUNet


# Dataset
def dataloader(path, pre_transforms, subject_ID = None):
    ''' Dataloader for inference
    path: path to the data directory
    pre_transforms: transforms to be applied to the data
    subject_ID: subject ID to be infered
    '''
    dataDir_img = path + 'images/'
    if subject_ID != None:
        images = sorted(glob(os.path.join(dataDir_img, str(subject_ID) + '*')))
    else:
        images = sorted(glob(os.path.join(dataDir_img, '*.nii*')))
    
    new_files = [
        {'image': img} for img in images[:]]
    images_ds = Dataset(
        data=new_files, transform=pre_transforms)
    return DataLoader(images_ds, batch_size=1, shuffle=False), images


def infer(args):
    if args.tta == 'yes':
        if args.tta_model==None:
            save_root = args.predictionsDir + '/' + args.datasetName + '/tta/' + args.model + \
            '_' + args.nepochs + 'e_' + args.data + '/' + args.tta_nepochs
            if args.norm_arc == 'deep':
                save_root = args.predictionsDir + '/' + args.datasetName + '/tta_deep/' + args.model + \
                '_' + args.nepochs + 'e_' + args.data + '/' + args.tta_nepochs
            elif args.norm_arc == 'deeper':
                save_root = args.predictionsDir + '/' + args.datasetName + '/tta_deeper/' + args.model + \
                '_' + args.nepochs + 'e_' + args.data + '/' + args.tta_nepochs
            elif args.norm_arc == 'deeper2x':
                save_root = args.predictionsDir + '/' + args.datasetName + '/tta_deeper2x/' + args.model + \
                '_' + args.nepochs + 'e_' + args.data + '/' + args.tta_nepochs

        else: 
            save_root = args.predictionsDir + '/' + args.datasetName + '/tta/' + args.tta_model[:20] + '/' + args.model + \
            '_' + args.nepochs + 'e_' + args.data + '/' + args.tta_nepochs
            if args.norm_arc == 'deep':
                save_root = args.predictionsDir + '/' + args.datasetName + '/tta_deep/' + args.tta_model[:20] + '/' + args.model + \
                '_' + args.nepochs + 'e_' + args.data + '/' + args.tta_nepochs
            elif args.norm_arc == 'deeper':
                save_root = args.predictionsDir + '/' + args.datasetName + '/tta_deeper/' + args.tta_model[:20] + '/' + args.model + \
                '_' + args.nepochs + 'e_' + args.data + '/' + args.tta_nepochs
            elif args.norm_arc == 'deeper2x':
                save_root = args.predictionsDir + '/' + args.datasetName + '/tta_deeper2x/' + args.tta_model[:20] + '/' + args.model + \
                '_' + args.nepochs + 'e_' + args.data + '/' + args.tta_nepochs

    else:
        if args.finetune == 'yes':
            if args.n_samples != '0':
                save_root = args.predictionsDir + '/' + args.datasetName + '/finetune/'+ args.n_samples +'samples_seed' + args.seed + '/' + args.model + \
                '_' + args.nepochs + 'e_' + args.data + '/' + args.finetune_epochs
            else: 
                save_root = args.predictionsDir + '/' + args.datasetName + '/finetune/'+ args.params + '/' + args.model + \
                '_' + args.nepochs + 'e_' + args.data + '/' + args.finetune_epochs
        else: 
            save_root = args.predictionsDir + '/' + args.datasetName + '/' + args.model + \
                '_' + args.nepochs + 'e_' + args.data + '/'

    if not osp.exists(save_root):
        os.makedirs(save_root)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pre_transforms = get_infer_transforms(args.SPATIAL_SIZE, args.roi_center)
    post_transforms = get_post_transforms(pre_transforms, args.thr)
    
    images_loader, images_name = dataloader(args.dataDir, pre_transforms, subject_ID = args.sub_ID)

    # Models
    if args.norm_arc == 'original':
        norm_model = Normalizer()
    elif args.norm_arc == 'deep':
        print('Deep normalization model')
        norm_model = Normalizer_Deep()
    elif args.norm_arc == 'deeper':
        print('Deeper normalization model')
        norm_model = Normalizer_Deeper()
    elif args.norm_arc == 'deeper2x':
        print('Deeper2x normalization model')
        norm_model = Normalizer_Deeper2x()

    segmenter = segmenter_model(args.CHANNELS, args.LABELS)
    
    if args.model == 'NormSeg':

        with torch.no_grad():
            for index, data in enumerate(images_loader):
                
                original_affine = data["image_meta_dict"]["affine"][0].numpy()
                name = os.path.split(images_name[index])[1]
                print(os.path.split(images_name[index]))

                if args.tta == 'yes':
                    if args.tta_model==None:
                        path_to_model = args.modelDir + 'tta/' + args.datasetName + '/' + name[:-7] + '_' + 'tta_' + args.tta_nepochs + 'e_'
                    else:
                        path_to_model = args.modelDir + 'tta/' + args.datasetName + '/' + args.tta_model + '_' + 'tta_' + args.tta_nepochs + 'e_'
                else:
                    path_to_model = args.modelDir

                print(path_to_model + 'normalization_model_' + args.nepochs + 'e_' + args.data + '.pt')
                norm_model.load_state_dict(torch.load(
                    os.path.join(path_to_model + 'normalization_model_' + args.nepochs + 'e_' + args.data + '.pt')))
                segmenter.load_state_dict(torch.load(
                    os.path.join(args.modelDir + 'segmenter_model_' + args.nepochs + 'e_' + args.data + '.pt')))
                
                norm_model.to(device)
                norm_model.eval()
                segmenter.to(device)
                segmenter.eval()

                input = (data["image"]).to(device)
                input = norm_model(input)
                data['pred'] = segmenter(input)[0]
                data['image'] = data['image'][0]

                for t in post_transforms:
                    data = t(data)

                nib.save(
                    nib.Nifti1Image(data['pred'][0].astype(np.uint8), original_affine), os.path.join(save_root, name))
                labels_to_int(os.path.join(save_root, name))

    else:

        with torch.no_grad():
            for index, data in enumerate(images_loader):
                original_affine = data["image_meta_dict"]["affine"][0].numpy()
                name = os.path.split(images_name[index])[1]

                if args.finetune == 'no':
                    model_name = os.path.join(args.modelDir + 'segmenter_model_only_' + args.nepochs + 'e_' + args.data + '.pt')
                    segmenter.load_state_dict(torch.load(model_name, map_location=device))
                    
                elif args.finetune == 'yes':
                    if args.n_samples != '0':
                        model_name = os.path.join(args.modelDir + 'finetune/' + args.datasetName + '/' + \
                                                    args.n_samples +'samples_seed' + args.seed + '_finetuned_' + args.finetune_epochs + \
                                                    'e_segmenter_model_only_' + args.nepochs + 'e_' + args.data + '.pt')
                        segmenter.load_state_dict(torch.load(model_name, map_location=device))

                    else:
                        if args.params == 'all':
                            model_name = os.path.join(args.modelDir + 'finetune/' + args.datasetName + '/' + \
                                                        name[:-7] + '_allParams_finetuned_' + args.finetune_epochs + \
                                                        'e_segmenter_model_only_' + args.nepochs + 'e_' + args.data + '.pt')
                            segmenter.load_state_dict(torch.load(model_name, map_location=device))

                        elif args.params == 'norm_layers':
                            model_name = os.path.join(args.modelDir + 'finetune/' + args.datasetName + '/' + \
                                                        name[:-7] + '_normLayers_finetuned_' + args.finetune_epochs + \
                                                        'e_segmenter_model_only_' + args.nepochs + 'e_' + args.data + '.pt')
                            segmenter.load_state_dict(torch.load(model_name, map_location=device))
                        elif args.params == 'input_block':
                            model_name = os.path.join(args.modelDir + 'finetune/' + args.datasetName + '/' + \
                                                        name[:-7] + '_inputBlock_finetuned_' + args.finetune_epochs + \
                                                        'e_segmenter_model_only_' + args.nepochs + 'e_' + args.data + '.pt')
                            segmenter.load_state_dict(torch.load(model_name, map_location=device))
                segmenter.to(device)
                segmenter.eval()
                print('Individual: ' + os.path.split(images_name[index])[1])
                print('Model: ' + model_name)
                input = (data["image"]).to(device)

                if args.tta == 'yes':
                    if args.tta_model==None:
                        path_to_model = args.modelDir + 'tta/' + args.datasetName + '/' + name[:-7] + '_' + 'tta_' + args.tta_nepochs + 'e_'
                        if args.norm_arc == 'deep':
                            path_to_model = args.modelDir + 'tta/' + args.datasetName + '/' + name[:-7] + '_deep' + '_' + 'tta_' + args.tta_nepochs + 'e_'
                        elif args.norm_arc == 'deeper':
                            path_to_model = args.modelDir + 'tta/' + args.datasetName + '/' + name[:-7] + '_deeper' + '_' + 'tta_' + args.tta_nepochs + 'e_'
                        elif args.norm_arc == 'deeper2x':
                            path_to_model = args.modelDir + 'tta/' + args.datasetName + '/' + name[:-7] + '_deeper2x' + '_' + 'tta_' + args.tta_nepochs + 'e_'

                    else:
                        path_to_model = args.modelDir + 'tta/' + args.datasetName + '/' + args.tta_model + '_' + 'tta_' + args.tta_nepochs + 'e_'
                        if args.norm_arc == 'deep':
                            path_to_model = args.modelDir + 'tta/' + args.datasetName + '/' + args.tta_model + '_deep' + '_' + 'tta_' + args.tta_nepochs + 'e_'
                        elif args.norm_arc == 'deeper':
                            path_to_model = args.modelDir + 'tta/' + args.datasetName + '/' + args.tta_model + '_deeper' + '_' + 'tta_' + args.tta_nepochs + 'e_'
                        elif args.norm_arc == 'deeper2x':
                            path_to_model = args.modelDir + 'tta/' + args.datasetName + '/' + args.tta_model + '_deeper2x' + '_' + 'tta_' + args.tta_nepochs + 'e_'

                    norm_model.load_state_dict(torch.load(
                                                        os.path.join(path_to_model + 'normalization_model_' + args.nepochs + 'e_' + args.data + '.pt'), map_location=device))
                    
                    print(os.path.join(path_to_model + 'normalization_model_' + args.nepochs + 'e_' + args.data + '.pt'))
                    norm_model.to(device)
                    norm_model.eval()
                    input = norm_model(input)
                    data['pred'] = segmenter(input)[0]
                else: 
                    data['pred'] = segmenter(input)[0]

                data['image'] = data['image'][0]
                for t in post_transforms:
                    data = t(data)
                
                nib.save(
                    nib.Nifti1Image(data['pred'][0].astype(np.uint8), original_affine), os.path.join(save_root, name))
                labels_to_int(os.path.join(save_root, name))
                print('Saved to: ', os.path.join(save_root, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataDir", type=str)
    parser.add_argument("-p", "--predictionsDir", default="./predictions/")
    parser.add_argument("-m", "--modelDir", default="./models/")
    parser.add_argument("--datasetName", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--nepochs", type=str)
    parser.add_argument("--tta", type=str, default = 'no')
    parser.add_argument("--sub_ID", type=str)
    parser.add_argument("--tta_model", type=str)
    parser.add_argument("--tta_nepochs", type=str)
    parser.add_argument("--thr", type=float)
    parser.add_argument("--roi_center", type=int, nargs=3)
    parser.add_argument("--datatype", type=str, default='tongue')
    parser.add_argument("--norm_arc", default='original')
    ## Finetuning
    parser.add_argument("--finetune", type=str, default = 'no')
    parser.add_argument("--params", default='all', type=str)
    parser.add_argument("--n_samples", type=str, default='0')
    parser.add_argument("--seed", type=str)
    parser.add_argument("--finetune_epochs", type=str)


    args = parser.parse_args()

    if args.datatype == "tongue":
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
            args.roi_center = (100, 295, 210)
        elif args.datasetName=='BeLong_T1w':
            args.roi_center = (100, 295, 210)
        elif args.datasetName[:4]=='EATT':
            args.roi_center = (100, 250, 90) 
        elif args.datasetName[:6]== 'Sydney':
            args.roi_center = (100, 250, 90)

    if args.datatype == 'prostate':
        args.SPATIAL_SIZE = [288, 288, 64]
        args.LABELS = {
            "prostate": 1,
            "background": 0,
        }
        args.roi_center = (220, 250, 32)

    args.CHANNELS = 1

    if args.datatype == 'organ':
        args.SPATIAL_SIZE = [192, 192, 64]
        args.LABELS = {
            "organ1": 1, "organ2": 2, "organ3":3,
            "background": 0,
        }
        # Using the center as roi
        args.roi_center = (144, 117, 61)

    infer(args)


if __name__ == '__main__':
    main()