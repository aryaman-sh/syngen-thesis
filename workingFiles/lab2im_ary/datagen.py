from sklearn.cluster import DBSCAN
import nibabel as nib
import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# My clustering algorithms

SCAN_PATH = '/scratch/itee/uqasha24/datasets/2023-08-22-aryaman/images/sub-10010.FatImaging_W.nii.gz'
SEG_PATH = '/scratch/itee/uqasha24/datasets/2023-08-22-aryaman/labels/sub-10010.FatImaging_W.nii.gz'
CLUSTER_TYPE = 'KMEANS' # ['DBSCAN' or 'KMEANS']
SCALING_FACTOR = 1024
CLUSTERS = 20
SCALING = False
OUTPUT_NAME = 'output.nii.gz'

# load scans
scan = nib.load(SCAN_PATH)
segmentation_mask = nib.load(SEG_PATH)

# get the data 
mri_data = scan.get_fdata()
mask_data = segmentation_mask.get_fdata()

# get anomaly indices and set all anomaly location to the mean
anomaly_indices = np.where(mask_data == 1)
anomaly_values = mri_data[anomaly_indices]
average_value = np.mean(anomaly_values)
mri_data[anomaly_indices] = average_value


# New scan where anomaly is just the anomaly values
mri_data_flattened = mri_data.reshape(-1, 1)

# CLUSTERING
if CLUSTER_TYPE=='KMEANS':
    # Define the number of clusters for k-means
    n_clusters = CLUSTERS  # Adjust as needed
    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # Fit the model to the flattened data
    kmeans.fit(mri_data_flattened)
    # Get cluster labels for each pixel in the flattened data
    cluster_labels = kmeans.labels_
elif CLUSTER_TYPE=='DBSCAN':
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(mri_data_flattened)
    cluster_labels = dbscan.labels_

# Now have k+1 clusters, one uniquely for the seg map, fixes the problem of having anomaly everywhere
clr = cluster_labels.reshape(mri_data.shape)
clr[anomaly_indices] = n_clusters
clr = clr.reshape(-1,1)

# Calculate the average pixel value for each cluster
cluster_average_values = [np.mean(mri_data_flattened[clr == i]) for i in range(n_clusters+1)]

# Assign cluster average values to all pixels in the cluster
for i in range(n_clusters+1):
    cluster_value = cluster_average_values[i]
    mri_data_flattened[clr == i] = cluster_value

# Reshape the modified flattened data back to the original shape
modified_mri_data = mri_data_flattened.reshape(mri_data.shape)
if SCALING:
    scaled_array = (modified_mri_data - modified_mri_data.min()) / (modified_mri_data.max() - modified_mri_data.min()) * SCALING_FACTOR
    modified_mri_data = np.round(scaled_array)

modified_scan = nib.Nifti1Image(modified_mri_data, scan.affine)
nib.save(modified_scan, OUTPUT_NAME)
