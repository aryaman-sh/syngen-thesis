from monai.apps.deepedit.transforms import (
    FindAllValidSlicesMissingLabelsd, NormalizeLabelsInDatasetd)
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Resized,
    Compose,
    RandCropByPosNegLabeld,
    RandFlipd,
    NormalizeIntensityd,
    SelectItemsd,
    ToTensord,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    Activationsd,
    AsDiscreted,
    EnsureTyped,
    SqueezeDimd,
    ToNumpyd,
    AddChanneld,
    SpatialCropd,
    Invertd,
    CenterSpatialCropd,
)
from monai.transforms import MapTransform, Resize
from typing import Optional, Sequence, Union
from monai.utils import InterpolateMode, ensure_tuple_rep
from monai.config import KeysCollection
from monai.data import MetaTensor
import numpy as np
import torch
import nibabel as nib

# For training
def labels_to_int(segmentation):
    nii_file = nib.load(segmentation)
    data = nii_file.get_fdata()
    new_header = nii_file.header.copy()
    new_header.set_data_dtype(np.int16)
    modified_nii_file = nib.Nifti1Image(data, nii_file.affine, new_header)
    nib.save(modified_nii_file, segmentation)

def get_train_transforms(spatial_size, labels):
    train_transforms = Compose([LoadImaged(keys=("image", "label"), ),
                                EnsureChannelFirstd(keys=("image", "label")),
                                Orientationd(
                                    keys=["image", "label"], axcodes="RAS"),
                                NormalizeIntensityd(
                                    keys="image", nonzero=True, channel_wise=True),
                                # data augmentation
                                RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=spatial_size,
        pos=1.,
        neg=0,
        num_samples=4,
        image_key="image",
        image_threshold=0,
    ),
        RandZoomd(
        keys=["image", "label"],
        min_zoom=0.9,
        max_zoom=1.2,
        mode=("trilinear", "nearest"),
        align_corners=(True, None),
        prob=0.15,
    ),
        RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
        RandGaussianSmoothd(
        keys=["image"],
        sigma_x=(0.5, 1.15),
        sigma_y=(0.5, 1.15),
        sigma_z=(0.5, 1.15),
        prob=0.15,
    ),
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
        RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(["image", "label"], spatial_axis=[2], prob=0.5),
        ToTensord(keys=("image", "label")),
        SelectItemsd(keys=("image", "label"))
    ])
    return train_transforms


def get_val_transforms(spatial_size, roi_center_location):
    val_transforms = Compose([LoadImaged(keys=("image", "label")),
                              EnsureChannelFirstd(keys=("image", "label")),
                              Orientationd(
        keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(
        keys="image", nonzero=True, channel_wise=True),
        SpatialCropd(keys=("image", "label"), roi_center=roi_center_location, 
                    roi_size = spatial_size), 
        ToTensord(keys=("image", "label")),
        SelectItemsd(keys=("image", "label"))]
    )
    return val_transforms


def get_eval_post_transforms(labels):
    post_transforms = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=len(labels))]
    return post_transforms


# For inference
def get_infer_transforms(spatial_size, roi_center_location):
    pre_transforms = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        SpatialCropd(keys="image", roi_center=roi_center_location,
                     roi_size=spatial_size), 
        EnsureTyped(keys="image")
    ])
    return pre_transforms


def get_post_transforms(pre_transforms, threshold = None):
    post_transforms = [
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True, threshold = threshold),
        Invertd(keys="pred", transform=pre_transforms, orig_keys="image"),
    ]
    return post_transforms
