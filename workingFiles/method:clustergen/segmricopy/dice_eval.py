"""
Give path to two nii images and prints the dice score. Two masks
"""

import numpy as np
import torch
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.transforms import LoadImaged, EnsureTyped, AddChanneld, Compose
from monai.transforms import Compose, LoadImaged, AddChanneld, EnsureTyped, AsDiscrete
from monai.metrics import DiceMetric
import torch


#print_config()
# Load your segmentation mask and predicted mask
# sub-10050.FatImaging_W.nii.gz, sub-10170.FatImaging_W.nii.gz  sub-10280.FatImaging_W.nii.gz
seg_mask_path = '/scratch/itee/uqasha24/synthetic-generalisation/fatdata_split/eval/labels/sub-10280.FatImaging_W.nii.gz'
pred_mask_path = '/scratch/itee/uqasha24/synthetic-generalisation/method:synthseg/SegMRI/pred_val/organ/segmenter_2e_organ/sub-10280.FatImaging_W.nii.gz'

# Define a dictionary for your data paths
data = [{'seg': seg_mask_path, 'pred': pred_mask_path}]

# Define MONAI transforms for loading and preprocessing the images
transforms = Compose([
    LoadImaged(keys=['seg', 'pred']),
    AddChanneld(keys=['seg', 'pred']),
    EnsureTyped(keys=['seg', 'pred'])
])

# Apply the transforms to load your images
transformed_data = transforms(data)
seg_mask_tensor = transformed_data[0]['seg'].unsqueeze(0)  # Add batch dimension
pred_mask_tensor = transformed_data[0]['pred'].unsqueeze(0)  # Add batch dimension

num_classes = 4  # Adjust based on your actual number of classes
to_one_hot_transform = Compose([
    AsDiscrete(to_onehot=True, n_classes=num_classes)
])

seg_mask_tensor_one_hot = to_one_hot_transform(seg_mask_tensor)
pred_mask_tensor_one_hot = to_one_hot_transform(pred_mask_tensor)

# Initialize DiceMetric for multi-class evaluation, including the background
dice_metric = DiceMetric(include_background=True, reduction='mean_channel')

# Compute Dice score within a no_grad context
with torch.no_grad():
    dice_score = dice_metric(pred_mask_tensor_one_hot, seg_mask_tensor_one_hot)

# dice_score now contains the mean Dice scores for each class
# If you want the overall mean across classes (excluding background), you can calculate that separately
dice_scores_numpy = dice_score.detach().cpu().numpy()  # Convert to NumPy array for easier manipulation
if num_classes > 1:
    mean_dice_excluding_background = dice_scores_numpy[1:].mean()  # Exclude background (class 0)
    print(f'Mean Dice Score (excluding background): {mean_dice_excluding_background}')
else:
    print(f'Dice Score: {dice_scores_numpy.item()}')