# STEP 1
"""
- ALL ORIGINAL LABELS ARE SAME CLUSTER

GIVEN A IMAGE AND LABEL CREATE CLUSTERS
"""
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans

train_names = ['10010'] # names of files
k_vals = [15, 25, 45] # Cluster values to generate

def create_clustered_nifti(original_nifti_path, label_map_path, output_path, n_clusters=10):
    # Load the original image and label map
    original_img = nib.load(original_nifti_path)
    label_map = nib.load(label_map_path)

    # Extract the data arrays
    original_data = original_img.get_fdata()
    label_data = label_map.get_fdata()

    # Find the maximum label in the label map
    max_label = np.max(label_data)

    # Find indices where label is zero
    zero_label_indices = np.where(label_data == 0)
    non_zero_label_indices = np.where(label_data != 0)

    # Extract data for clustering from the zero-label regions
    data_for_clustering = original_data[zero_label_indices]

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters)
    clustered_labels = kmeans.fit_predict(data_for_clustering.reshape(-1, 1))

    # Offset the cluster labels to avoid overlap with existing labels
    offset_clustered_labels = clustered_labels + max_label + 1

    # Replace zero-labeled areas in the original image with the offset cluster labels
    new_data = np.copy(original_data)
    new_data[zero_label_indices] = offset_clustered_labels
    new_data[non_zero_label_indices] = label_data[non_zero_label_indices]
    print(np.unique(new_data))
    # Create a new NIfTI image
    new_img = nib.Nifti1Image(new_data, original_img.affine, original_img.header)

    # Save the new image
    nib.save(new_img, output_path)

for tn in train_names:
    for kv in k_vals:
        # Load your NIfTI image and label map
        print(f'{tn}, {str(kv)}')
        nii_image_path = f'../../fatdata/images/sub-{tn}.FatImaging_W.nii.gz' 
        label_map_path = f'../../fatdata/labels/sub-{tn}.FatImaging_W.nii.gz'  
        # Example usage
        create_clustered_nifti(nii_image_path, label_map_path, f'./step1data/{tn}_{str(kv)}.nii.gz', n_clusters=kv)