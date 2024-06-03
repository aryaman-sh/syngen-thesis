"""
Generating labels
pretty straightforwards
no assumptions
"""
"""
For estimating generation labels, generation classes and stuff
"""
"""
CHANGELOG:
Now since clusters are random need to loop over all files then do it for all

"""

import nibabel as nib
import numpy as np
import os
import argparse


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


def main():
    files = os.listdir('./step1data/')
    for file in files:
        filename = file
        image_path = f'./step1data/{filename}'
        image = nib.load(image_path)
        image_data = image.get_fdata()

        generation_labels = get_generation_labels(image_data)
        output_labels = get_output_labels(image_data)
        generation_classes = get_generation_classes(generation_labels)

        generation_labels = generation_labels.astype(np.int32)
        output_labels = output_labels.astype(np.int32)
        generation_classes = generation_classes.astype(np.int32)

        img_id = filename.replace(".nii.gz", "")
        np.save(f'./step2data/generation_labels_{img_id}.npy', generation_labels)
        np.save(f'./step2data/output_labels_{img_id}.npy', output_labels)
        np.save(f'./step2data/generation_classes_{img_id}.npy', generation_classes)
        print(f"Step 2 complete for {img_id} ")


if __name__ == "__main__":
    os.mkdir('./step2data')
    main()
