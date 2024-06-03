"""
This intends to cluster voxels into k clusters and produce a clustered label map where each label "class" is the mean of voxel values in that cluster

TODO: Currently generated based on clustering on a single image, think about if clustering should be uniform throughout multiple samples or if each should be individually clustered
- Also look at post processing to maybe clean up the clusters a bit, if that helps
"""

import nibabel as nib
from sklearn.cluster import KMeans
import numpy as np
from scipy.ndimage import binary_opening, binary_closing

def cluster_voxels_and_create_mask(input_file_path, output_file_path, n_clusters):
    # Load the scan using nibabel
    scan = nib.load(input_file_path)
    data = scan.get_fdata()
    
    # Reshape data to 2D array (voxels x features)
    original_shape = data.shape
    data_reshaped = data.reshape(-1, 1)  # Reshaping to a 2D array

    # Apply KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data_reshaped)

    # Create segmentation mask
    segmentation_mask = kmeans.labels_.reshape(original_shape)

    # Create a new Nifti image for the segmentation mask
    mask_image = nib.Nifti1Image(segmentation_mask, affine=scan.affine)

    # Save the segmentation mask image
    nib.save(mask_image, output_file_path)

def postprocess_segmentation(segmentation_mask):
    # Apply binary opening and then closing
    processed_mask = binary_opening(segmentation_mask, structure=np.ones((3,3,3)))
    processed_mask = binary_closing(processed_mask, structure=np.ones((3,3,3)))
    return processed_mask

if __name__ == "__main__":
    input_file_path = "../../fatdata/images/sub-10010.FatImaging_W.nii.gz"
    output_file_path = "/scratch/itee/uqasha24/synthetic-generalisation/method:synthseg/experiments/2/clusteredData/16/cluster16.nii.gz"
    n_clusters = 16

    cluster_voxels_and_create_mask(input_file_path, output_file_path, n_clusters)