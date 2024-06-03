"""
Generating labels
pretty straightforwards
no assumptions
"""
"""
For estimating generation labels, generation classes and stuff
"""

import nibabel as nib
import numpy as np
import os

def get_generation_labels(image_data):
    unique_pixel_values = np.unique(image_data)
    return unique_pixel_values

def get_output_labels(image_data):
    # currently i know that first three are anomaly labels
    # TODO: make it generalised and dynamic
    unique_pixel_values = np.unique(image_data)
    new_arr = np.zeros_like(unique_pixel_values)
    new_arr[:3] = unique_pixel_values[:3]
    return new_arr

def get_generation_classes(generation_labels):
    gen_arr = np.arange(generation_labels.max())
    return gen_arr

if __name__ == "__main__":
    train_names = ['10010']
    k_vals = [15, 25, 45]
    
    for tn in train_names:
        for kv in k_vals:
            image_path = f'./step1data/{tn}_{kv}.nii.gz'
            image = nib.load(image_path)
            image_data = image.get_fdata()

            generation_labels = get_generation_labels(image_data)
            output_labels = get_output_labels(image_data)
            generation_classes = get_generation_classes(generation_labels)

            generation_labels = generation_labels.astype(np.int32)
            output_labels = output_labels.astype(np.int32)
            generation_classes = generation_classes.astype(np.int32)

            np.save(f'./step2data/generation_labels_{tn}_{kv}.npy', generation_labels)
            np.save(f'./step2data/output_labels_{tn}_{kv}.npy', output_labels)
            np.save(f'./step2data/generation_classes_{tn}_{kv}.npy', generation_classes)