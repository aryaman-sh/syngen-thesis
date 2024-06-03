import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import nibabel as nib


def input_check(epoch, model, data, normalized_input, input, save_root):
    save_root = save_root + '/' + model + '/'

    if not osp.exists(save_root):
        os.makedirs(save_root)

    # Selecting a slice
    norm_input = normalized_input[0][0][50, :, :] # 20 for training
    input = input[0][0][50, :, :] # 20 for training

    # Settings
    fig = plt.figure(figsize=(30, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(np.rot90(norm_input), cmap='gray',
               vmax=np.max(norm_input),  vmin=np.min(norm_input))

    ax2 = fig.add_subplot(1, 2, 1)
    ax2.imshow(np.rot90(input), cmap='gray',
               vmax=np.max(input),  vmin=np.min(input))
    plt.tight_layout()
    plt.savefig(save_root + data + '_check_epoch' + str(epoch), dpi=300)


def segmentation_visualization(files, n_predictions, index, save = False, tta_model = None, vmax = 50, slice_index = 100):
    data = {}
    
    for i in range(n_predictions):
        data['prediction' + str(i + 1)] = nib.load(
            files['prediction' + str(i + 1)][index]).get_fdata()
        data['img' + str(i + 1)] = nib.load(
            files['img' + str(i + 1)][index]).get_fdata()
        data['groundtruth' + str(i + 1)] = nib.load(
            files['groundtruth' + str(i + 1)][index]).get_fdata()
    dataset = files['img' + str(i + 1)][index].split('/')
    fig, axs = plt.subplots(n_predictions, 2,
                            figsize=(3*n_predictions, 6))
        
    if n_predictions ==1:
            axs[0].imshow(
                np.rot90(data['img' + str(i + 1)][slice_index, :, :]), cmap='gray', vmax = vmax)
            axs[0].imshow(
                np.rot90(data['prediction' + str(i + 1)][slice_index, :, :]), cmap='magma', alpha=.5, vmax = 5)
            axs[1].imshow(
                np.rot90(data['img' + str(i + 1)][slice_index, :, :]), cmap='gray', vmax = vmax)
            axs[1].imshow(
                np.rot90(data['groundtruth' + str(i + 1)][slice_index, :, :]), cmap='magma', alpha=.5, vmax = 5)
    else:
        for i in range(n_predictions):
            axs[i, 0].imshow(
                np.rot90(data['img' + str(i + 1)][slice_index, :, :]), cmap='gray', vmax = vmax)
            axs[i, 0].imshow(
                np.rot90(data['prediction' + str(i + 1)][slice_index, :, :]), cmap='magma', alpha=.5, vmax = 5)
            axs[i, 1].imshow(
                np.rot90(data['img' + str(i + 1)][slice_index, :, :]), cmap='gray', vmax = vmax)
            axs[i, 1].imshow(
                np.rot90(data['groundtruth' + str(i + 1)][slice_index, :, :]), cmap='magma', alpha=.5, vmax = 5)
    

    plt.tight_layout()
    if save == True:
        if tta_model == None:
            directory = './QC/' + dataset[2] + '_' + dataset[3] + '/' 
        else:
            directory = './QC/' + dataset[2] + '_' + dataset[3] + '_' + tta_model + '/' 
        if not osp.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + dataset[5] + '.png', transparent = True,
                   bbox_inches = 'tight', dpi = 300)

        
def segmentation_visualization_all(files, n_predictions, index, save = False, vmax = 50):
    data = {}
    
    for i in range(n_predictions):
        data['prediction' + str(i + 1)] = nib.load(
            files['prediction' + str(i + 1)][index]).get_fdata()
        data['img' + str(i + 1)] = nib.load(
            files['img' + str(i + 1)][index]).get_fdata()
        data['groundtruth' + str(i + 1)] = nib.load(
            files['groundtruth' + str(i + 1)][index]).get_fdata()
    dataset = files['img' + str(i + 1)][index].split('/')
    fig, axs = plt.subplots(1, n_predictions + 1,
                            figsize=(3*(n_predictions + 1), 3))
    if n_predictions ==1:
            axs[0].imshow(
                np.rot90(data['img' + str(i + 1)][50, :, :]), cmap='gray', vmax = vmax)
            axs[0].imshow(
                np.rot90(data['groundtruth' + str(i + 1)][50, :, :]), cmap='magma', alpha=.5, vmax = 5)
            axs[1].imshow(
                np.rot90(data['img' + str(i + 1)][50, :, :]), cmap='gray', vmax = vmax)
            axs[1].imshow(
                np.rot90(data['prediction' + str(i + 1)][50, :, :]), cmap='magma', alpha=.5, vmax = 5)

    else:
        for i in range(n_predictions + 1):
            if i == 0:
                axs[i].imshow(
                    np.rot90(data['img' + str(i + 1)][50, :, :]), cmap='gray', vmax = vmax)
                axs[i].imshow(
                    np.rot90(data['groundtruth' + str(i + 1)][50, :, :]), cmap='magma', alpha=.5, vmax = 5)
                axs[i].title.set_text('atlas')
            else:
                axs[i].imshow(
                    np.rot90(data['img' + str(i)][50, :, :]), cmap='gray', vmax = vmax)
                axs[i].imshow(
                    np.rot90(data['prediction' + str(i)][50, :, :]), cmap='magma', alpha=.5, vmax = 5)
                axs[i].title.set_text('prediction_' + str(i))
    

    plt.tight_layout()
    if save == True:
        directory = './QC/all/' + dataset[2] + '_' + dataset[3] + '/' 
        if not osp.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + dataset[5] + '.png', transparent = True,
                   bbox_inches = 'tight', dpi = 300)