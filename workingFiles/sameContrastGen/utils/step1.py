# STEP 1
"""
- ALL ORIGINAL LABELS ARE SAME CLUSTER

GIVEN A IMAGE AND LABEL CREATE CLUSTERS

Changelog 
- Select cluster value randomly in the range (20, 40)

"""
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
import random
import time
import os
from glob import glob
import argparse

# train_names = ['10010','10100', '10360', '10430', '10780', '10930', '12400',
#                 '13660', '14740', '15120', '15280', '17820', '19830', '44170', '52220'] # names of files

def create_clustered_nifti(original_nifti_path, label_map_path, output_path, n_clusters):
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

# start_time = time.time()
# for tn in train_names: # For each training image
#     for run in range(20): # for 20 runs for each image
#         kv = random.randint(20, 50) # random k value
#         # Load your NIfTI image and label map
#         print(f'Generating for {tn}, for kv {kv} , run: {run}')
#         nii_image_path = f'/scratch/itee/uqasha24/thesis2024/data/organseg/images/sub-{tn}.FatImaging_W.nii.gz'
#         label_map_path = f'/scratch/itee/uqasha24/thesis2024/data/organseg/labels/sub-{tn}.FatImaging_W.nii.gz'
#         # Example usage
#         st1 = time.time()
#         create_clustered_nifti(nii_image_path, label_map_path, f'./step1data/{tn}_{str(kv)}_r{run+1}.nii.gz', n_clusters=kv)
#         print(f"Took {st1 - time.time()}")
#
# end_time = time.time()
# print(f" Time taken: {start_time - end_time} Seconds")

def main(dir_path, k_min, k_max, num_gen):
    # Load images and labels
    os.mkdir("./step1data")
    dataDir_img = os.path.join(dir_path, "images")
    dataDir_seg = os.path.join(dir_path, "labels")
    images = sorted(glob(os.path.join(dataDir_img, '*.nii*')))
    segs = sorted(glob(os.path.join(dataDir_seg, '*.nii*')))
    for i in range(len(images)):
        assert images[i].split('/')[2] == segs[i].split('/')[2]
    print('All file names are correct')
    samples = len(images)
    # Data prep
    files = [
        {'image': img, "label": seg} for img, seg in zip(images, segs)
    ]

    start_time = time.time()
    for filepair in files:
        img = filepair['image']
        label = filepair['label']

        for run in range(num_gen):
            run = run+1
            kv = random.randint(k_min, k_max)
            print(f'Generating for {img}, for kv {kv} , run: {run}')
            st1 = time.time()
            create_clustered_nifti(img, label, f'./step1data/{os.path.splitext(os.path.basename(img))[0]}_{str(kv)}_r{run}.nii.gz',n_clusters=kv)
            print(f"Took {time.time()-st1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1")
    parser.add_argument("--data_dir", help="generation data dir")
    parser.add_argument("--k_min", type=int)
    parser.add_argument("--k_max", type=int)
    parser.add_argument("--num_gen", type=int)

    args = parser.parse_args()
    main(args.data_dir, args.k_min, args.k_max, args.num_gen)

