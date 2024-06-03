import monai.metrics as metrics
import torch
import nibabel as nib
import os
import numpy as np
from glob import glob
from monai.networks.utils import one_hot


def load_data(predictionsDir, dataDir, predictions, atlas = False):
    """Generate lists with original images, ground-truth segmentations, and predicted segmentations filenames.

    Args:
        predictionsDir (str): path to predictions directory
        dataDir (list): list with paths to data directory
        predictions (list): list with the conditions
    Returns:
        images (list): List of the image filenames (number_of_conditions, samples)
        segmentations (list): List of the segmentation filenames (number_of_conditions, samples)
        predicted_segmentations (list): List of the predicted_segmentation filenames (number_of_conditions, samples)

    """
    segmentations = []
    images = []
    predicted_segmentations = []
    
    
    if len(dataDir)== 1 and len(predictions)!=1:
        dataDir = dataDir *  len(predictions)
        
    for i in range(len(predictions)):

        img_path = dataDir[i] + 'images/'
        seg_path = dataDir[i] + 'labels/'

        images.append(sorted(glob(os.path.join(img_path, '*.nii.gz'))))
        if atlas == True:
            segmentations.append(sorted(glob(os.path.join(seg_path, '*.nii.gz'))) * len(sorted(glob(os.path.join(img_path, '*.nii.gz')))))
        else:
            segmentations.append(sorted(glob(os.path.join(seg_path, '*.nii.gz'))))
        
        predicted_segmentations.append(sorted(glob(os.path.join(predictionsDir, predictions[i] + '/*'))))

    return images, segmentations, predicted_segmentations


def load_data_all(predictionsDirs, dataDir, atlas = False):
    """Generate lists with original images, ground-truth segmentations, and predicted segmentations filenames.

    Args:
        predictionsDirs (list): paths to prediction directories
        dataDir (list): list with paths to data directory
    Returns:
        images (list): List of the image filenames (number_of_conditions, samples)
        segmentations (list): List of the segmentation filenames (number_of_conditions, samples)
        predicted_segmentations (list): List of the predicted_segmentation filenames (number_of_conditions, samples)

    """
    segmentations = []
    images = []
    predicted_segmentations = []
    
    
    if len(dataDir)== 1 and len(predictionsDirs)!=1:
        dataDir = dataDir *  len(predictionsDirs)
        
    for i in range(len(predictionsDirs)):

        img_path = dataDir[i] + 'images/'
        seg_path = dataDir[i] + 'labels/'

        images.append(sorted(glob(os.path.join(img_path, '*.nii.gz'))))
        if atlas == True:
            segmentations.append(sorted(glob(os.path.join(seg_path, '*.nii.gz'))) * len(sorted(glob(os.path.join(img_path, '*.nii.gz')))))
        else:
            segmentations.append(sorted(glob(os.path.join(seg_path, '*.nii.gz'))))
        
        predicted_segmentations.append(sorted(glob(predictionsDirs[i] +  '/*')))

    return images, segmentations, predicted_segmentations


def data_prep(score, imgs, groundtruths, predictions, dataset, labels, instance = 'single'):
    """Determine the score for predicted segmentations given the ground-truth data.

    Args:
        score (str): metric to be used 
        imgs (list): List of the image filenames (number_of_conditions, samples)
        groundtruths (list): List of the segmentation filenames (number_of_conditions, samples)
        predictions (list): List of the predicted_segmentation filenames (number_of_conditions, samples)
        dataset (str): 'train' or 'validation' set
        labels (dict): dictionary with segmentation labels
    Return:
        metrics_dict (dict): dictionary with keys being conditions and metric name, and values are list 
                            with individuals' scores and metric names
        files (dict): dictionary with keys being type of image, and values are lists with files paths
    """

    files = {}
    metrics_dict = {}

    if len(imgs) == 1 and len(predictions) != 1:
        imgs = [imgs] * len(predictions)
        groundtruths = [groundtruths] * len(predictions)

    samples = len(predictions[0])

    if dataset == 'train':
        for i in range(len(predictions)):
            files['img' + str(i + 1)] = [img for img in imgs[i][:int(samples*.95)]]
            files['groundtruth' +
                  str(i + 1)] = [seg for seg in groundtruths[i][:int(samples*.95)]]
            files['prediction' +
                  str(i + 1)] = [pred for pred in predictions[i][:int(samples*.95)]]
            # Condition
            names = files['prediction' + str(i + 1)][0].split('/')
            condition = names[3]
            if condition=='tta':
                condition = condition + '_' + names[4] + '_' + names[5] + '_' + names[6] + 'e'
            metrics_dict[condition] = []

    elif dataset == 'validation':
        for i in range(len(predictions)):
            files['img' + str(i + 1)] = [img for img in imgs[i][int(samples*.95):]]
            files['groundtruth' +
                  str(i + 1)] = [seg for seg in groundtruths[i][int(samples*.95):]]
            files['prediction' +
                  str(i + 1)] = [pred for pred in predictions[i][int(samples*.95):]]
            # Condition
            names = files['prediction' + str(i + 1)][0].split('/')
            condition = names[3]
            if condition=='tta':
                condition = condition + '_' + names[4] + '_' + names[5] + '_' + names[6] + 'e'
            metrics_dict[condition] = []
            
    else:
        for i in range(len(predictions)):
            files['img' + str(i + 1)] = [img for img in imgs[i][:]]
            files['groundtruth' +
                  str(i + 1)] = [seg for seg in groundtruths[i][:]]
            files['prediction' +
                  str(i + 1)] = [pred for pred in predictions[i][:]]
            # Condition
            names = files['prediction' + str(i + 1)][0].split('/')
            condition = names[3]
            if condition=='tta' or condition=='tta_deep' or condition=='tta_deeper':
                condition = condition + '_' + names[4] + '_' + names[5] + 'e' 
                if instance != 'single':
                    condition = condition + '_' + names[6]
            metrics_dict[condition] = []

    if score == 'dice':
        dice = metrics.DiceMetric(include_background=True, reduction="mean")
        samples = len(files['img1'])
        for j in range(len(predictions)):
            names = files['prediction' + str(j + 1)][0].split('/')
            condition = names[3]
            if condition=='tta' or condition=='tta_deep' or condition=='tta_deeper':
                condition = condition + '_' + names[4] + '_' + names[5] + 'e' 
                if instance != 'single':
                    condition = condition + '_' + names[6]
               
            for i in range(samples):
                prediction_name = files['prediction' + str(j + 1)][i].split('/')
                gt_name = files['groundtruth' + str(j + 1)][i].split('/')
                assert prediction_name[-1] == gt_name[-1]
                prediction = one_hot(torch.tensor(np.array([nib.load(
                    files['prediction' + str(j + 1)][i]).get_fdata()])), num_classes=len(labels), dim=0)
                label = one_hot(torch.tensor(np.array([nib.load(
                    files['groundtruth' + str(j + 1)][i]).get_fdata()])), num_classes=len(labels), dim=0)
                dice(prediction, label)
                score_seg = dice.aggregate().item()
                metrics_dict[condition].append(score_seg)
                dice.reset()

        metrics_dict['Metric'] = samples * ['DICE']

    return metrics_dict, files