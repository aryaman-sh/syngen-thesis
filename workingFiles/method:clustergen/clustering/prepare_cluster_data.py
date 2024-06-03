"""
Run this to prepare data for a cluster. Outputs results as npy files
"""

import nibabel as nib
import numpy as np

image_path = '/scratch/itee/uqasha24/synthetic-generalisation/cluster_test.nii.gz'
image = nib.load(image_path)
image_data = image.get_fdata()

unique_pixel_values = np.unique(image_data)

generation_labels = unique_pixel_values
output_labels = unique_pixel_values
generation_classes = unique_pixel_values

np.save('generation_labels.npy', generation_labels)
np.save('output_labels.npy', output_labels)
np.save('generation_classes.npy', generation_classes)

