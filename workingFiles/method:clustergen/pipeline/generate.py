"""
Synthetic data generator

Creates temp directories for all files.
"""

import argparse
import os
import glob
import tempfile
import atexit   
import shutil
import random
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans


def create_clustered_nifti(original_nifti_path, label_map_path, output_path, n_clusters=10):
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
    
    new_data = new_data.astype(np.int32)
    #print(np.unique(new_data))

    # Create a new NIfTI image
    new_img = nib.Nifti1Image(new_data, original_img.affine, original_img.header)

    # Save the new image
    nib.save(new_img, output_path)


def clusterHandler(gen_files, kmin, kmax, rep, temp_dir):
    """
    Given image, label pairs, adds data to a temporary directory, all clustered data
    k is randomly sampled
    """
    cluster_register = [] # a dataset of (path to image, cluster value)

    for data in gen_files:
        k_vals = random.sample(range(kmin, kmax), rep)
        namingBase = os.path.basename(data['image']).replace(".nii.gz", "")
        for k in k_vals:
            create_clustered_nifti(data['image'], data['label'], f'{temp_dir}/{namingBase}_{k}.nii.gz', k) 
            cluster_register.append((data['image'], k))
    return cluster_register

def cleanup_temp_dir():
    shutil.rmtree(temp_dir)
    print(f"Temporary directory {temp_dir} has been deleted.")

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

def generate_labels(cluster_register, temp_dir):
    for data in cluster_register:
        image_path = f'{temp_dir}/{os.path.basename(data[0]).replace(".nii.gz", "")}_{data[1]}.nii.gz'
        image = nib.load(image_path)
        image_data = image.get_fdata()
        generation_labels = get_generation_labels(image_data)
        output_labels = get_output_labels(image_data)
        generation_classes = get_generation_classes(generation_labels)

        generation_labels = generation_labels.astype(np.int32)
        output_labels = output_labels.astype(np.int32)
        generation_classes = generation_classes.astype(np.int32)
        np.save(f'{temp_dir}/{os.path.basename(data[0]).replace(".nii.gz", "")}_{data[1]}_generation_labels.npy', generation_labels)
        np.save(f'{temp_dir}/{os.path.basename(data[0]).replace(".nii.gz", "")}_{data[1]}_output_labels.npy', output_labels)
        np.save(f'{temp_dir}/{os.path.basename(data[0]).replace(".nii.gz", "")}_{data[1]}_generation_classes.npy', generation_classes)

if __name__ == "__main__":
    """
    Arguments
    - dataDir: data directory
    - clusterMin: When Clustering with multiple clusters, defines the min value for k
    - clusterMax: When clustering with multiple clusters, defines the max value for k
    For single cluster values i.e. using one k value, make sure ClusterMin == CluasterMax
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", type=str) # Data directory NOTE: (must end with /)
    parser.add_argument("--clusterMin", type=int, default=11) # Min num for clustering 
    parser.add_argument("--clusterMax", type=int, default=15) # Max for clustering
    parser.add_argument("--clusterRep", type=int, default = 1) # How many times is a single image used for clustering
    args = parser.parse_args()
    #atexit.register(cleanup_temp_dir)
    # Load images
    dataDir_img = args.dataDir + 'images/'
    dataDir_seg = args.dataDir + 'labels/'
    images = sorted(glob.glob(os.path.join(dataDir_img, '*.nii*')))
    segs = sorted(glob.glob(os.path.join(dataDir_seg, '*.nii*')))
    gen_files = [{'image': img, "label": seg} for img, seg in zip(images[:], segs[:])]
    
    # Perform clustering
    temp_dir = tempfile.mkdtemp(dir=os.getcwd())
    print(f"Created tempdir {temp_dir}")
    cluster_register = clusterHandler(gen_files, args.clusterMin, args.clusterMax, args.clusterRep, temp_dir)
    
    # Generate labels
    # cluster register contains image path , k value
    # Now this image and k value are used to reference clustered image 
    generate_labels(cluster_register, temp_dir)

    # Execute prior estimation and generation
    # Note this script does prior generation and estimation image wise and not data wise
    
