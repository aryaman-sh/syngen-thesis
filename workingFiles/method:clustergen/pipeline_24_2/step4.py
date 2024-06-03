"""
Generation


https://github.com/BBillot/lab2im/blob/master/bibtex.bib
Copyright 2020 Benjamin Billot
"""

#temp fix somehow fixes path stuff
# run this from inside the tutorials directory
import sys
sys.path.append('../')

import os
import time
from ext.lab2im import utils
from ext.lab2im.image_generator import ImageGenerator
import tempfile
import shutil

train_names = ['10280','10150', '10020', '10250', '10100'] # names of files
k_vals = [12,14,16,18,20,22,24] # Cluster values to generate
        
for run in range(5):
    for tn in train_names:
        for kv in k_vals:

            with tempfile.TemporaryDirectory() as tempdir1:
                # transfer labels to tempdir
                shutil.copy(f'/scratch/itee/uqasha24/thesis2024/samplingBased/pl1/step1data/{tn}_{kv}_r{run+1}.nii.gz', tempdir1)
                # label map to generate images from
                path_label_map = tempdir1

                # general parameters
                n_examples = 7
                result_dir = f'/scratch/itee/uqasha24/thesis2024/samplingBased/pl1/step4data/'
                output_shape = None  # shape of the output images, obtained by randomly cropping the generated images

                # specify structures from which we want to generate
                generation_labels = f'/scratch/itee/uqasha24/thesis2024/samplingBased/pl1/step2data/generation_labels_{tn}_{kv}_r{run+1}.npy'
                # specify structures that we want to keep in the output label maps
                output_labels = f'/scratch/itee/uqasha24/thesis2024/samplingBased/pl1/step2data/output_labels_{tn}_{kv}_r{run+1}.npy'
                # we regroup structures into K classes, so that they share the same distribution for image generation
                generation_classes = f'/scratch/itee/uqasha24/thesis2024/samplingBased/pl1/step2data/generation_classes_{tn}_{kv}_r{run+1}.npy'

                # We specify here that we type of prior distributions to sample the GMM parameters.
                # By default prior_distribution is set to 'uniform', and in this example we want to change it to 'normal'.
                prior_distribution = 'normal'
                # We specify here the hyperparameters of the prior distributions to sample the means of the GMM.
                # As these prior distributions are Gaussians, they are each controlled by a mean and a standard deviation.
                # Therefore, the numpy array pointed by prior_means is of size (2, K), where K is the nummber of classes specified in
                # generation_classes. The first row of prior_means correspond to the means of the Gaussian priors, and the second row
                # correspond to standard deviations.
                prior_means = f'/scratch/itee/uqasha24/thesis2024/samplingBased/pl1/step3data/prior_means_{tn}_{kv}_r{run+1}.npy'
                # same as for prior_means, but for the standard deviations of the GMM.
                prior_stds = f'/scratch/itee/uqasha24/thesis2024/samplingBased/pl1/step3data/prior_stds_{tn}_{kv}_r{run+1}.npy'

                ########################################################################################################

                # instantiate BrainGenerator object
                brain_generator = ImageGenerator(labels_dir=path_label_map,
                                                generation_labels=generation_labels,
                                                output_labels=output_labels,
                                                generation_classes=generation_classes,
                                                prior_distributions=prior_distribution,
                                                prior_means=prior_means,
                                                prior_stds=prior_stds,
                                                output_shape=output_shape)

                # create result dir
                utils.mkdir(result_dir)

                for n in range(1, n_examples + 1):

                    # generate new image and corresponding labels
                    start = time.time()
                    im, lab = brain_generator.generate_image()
                    end = time.time()
                    print('generation {0:d} took {1:.01f}s'.format(n, end - start))

                    # save output image and label map
                    utils.save_volume(im, brain_generator.aff, brain_generator.header,
                                    os.path.join(result_dir, f'images/{tn}_{kv}_r{run+1}_%s.nii.gz' % n))
                    utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                                    os.path.join(result_dir, f'labels/{tn}_{kv}_r{run+1}_%s.nii.gz' % n))
    

    