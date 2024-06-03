"""
Author: Thom + ChatGPT
Date: July 1, 2023
This script resizes a 3D NIFTI image file to a target shape by either padding or de-padding equally along each dimension.
"""

import argparse
import nibabel as nib
import numpy as np
from glob import glob
import os

def resize_image(path, target_shape):
    """
    Resize a 3D NIFTI image file to a target shape by either padding or de-padding.

    Parameters
    ----------
    input_image : str
        Path to the input NIFTI image.
    target_shape : numpy.ndarray
        Target shape after resizing as a 3-element numpy array.

    Raises
    ------
    AssertionError
        If the input image is not 3D.
    ValueError
        If the target shape is not valid.
    """
    dataDir_img = path 
    print(glob(os.path.join(dataDir_img, '*.nii*')))
    images = sorted(glob(os.path.join(dataDir_img, '*.nii*')))
    print(images)
    
    for image in images:
        # Load your NIFTI file
        nii = nib.load(image)

        # Extract the image data array
        data = nii.get_fdata()

        # Print the original size of the image
        print(f'Original image shape: {data.shape}')

        # Ensure the original image is 3D
        assert data.ndim == 3, "Input image must be 3D."

        # Check target shape validity
        if (target_shape <= 0).any():
            raise ValueError("All dimensions of target shape must be greater than 0.")

        # Calculate the padding or slicing dimensions
        current_shape = np.array(data.shape)
        resize_dims = target_shape - current_shape

        if (resize_dims < 0).any():
            # De-pad
            print('Depadding image...')
            remove_slices = [slice(-resize_dims[i] // 2, resize_dims[i] // 2) if resize_dims[i] < 0 else slice(None) for i in range(3)]
            resized_data = data[remove_slices[0], remove_slices[1], remove_slices[2]]
            operation = 'depadded'
        else:
            # Pad
            print('Padding image...')
            pad_width = [(resize_dims[i] // 2, (resize_dims[i] // 2) + (resize_dims[i] % 2)) for i in range(3)]
            resized_data = np.pad(data, pad_width, mode='constant', constant_values=0)
            operation = 'padded'

        # Print the size of the resized image
        print(f'Resized image shape: {resized_data.shape}')

        # Create a new NIFTI file with the resized data
        # Modify the affine matrix to account for the resizing in each dimension
        affine = nii.affine
        if operation == 'padded':
            affine[:3,3] -= np.array([pad_width[i][0] for i in range(3)]) * nii.header.get_zooms()
        else:  # operation == 'depadded'
            for i in range(3):
                if remove_slices[i].start is not None:
                    affine[:3,3][i] += remove_slices[i].start * nii.header.get_zooms()[i]
                else:
                    affine[:3,3][i] += 0

        resized_nii = nib.Nifti1Image(resized_data, affine)

        # Construct the output image name
        output_image = image.replace('.nii.gz', f'_{operation}_to_{target_shape[0]}_{target_shape[1]}_{target_shape[2]}.nii.gz')

        # Save the new NIFTI file
        nib.save(resized_nii, output_image)
        print(f'Saved {operation} image as {output_image}')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize a 3D NIFTI image to a target shape.")
    parser.add_argument("--path", type=str, help="Path to the input NIFTI image.")
    parser.add_argument("--target_shape", type=int, nargs=3, help="Target shape after resizing as three integers.")
    args = parser.parse_args()
    resize_image(args.path, np.array(args.target_shape))