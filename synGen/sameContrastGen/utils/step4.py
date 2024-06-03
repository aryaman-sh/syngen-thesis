"""
Generation


https://github.com/BBillot/lab2im/blob/master/bibtex.bib
Copyright 2020 Benjamin Billot
"""

#temp fix somehow fixes path stuff
# run this from inside the tutorials directory
import sys
#sys.path.append('../')

import os
import time
from ext.lab2im import utils
from ext.lab2im.image_generator import ImageGenerator
import tempfile
import shutil
import argparse

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Step 4")
    parser.add_argument("--data_dir", help="generation data dir")
    parser.add_argument("--out_dir", help="synth data output dir")
    args = parser.parse_args()

    os.mkdir(args.out_dir)
    os.mkdir(f'{args.out_dir}/images')
    os.mkdir(f'{args.out_dir}/labels')

    files = os.listdir('./step1data/')
            
    for file in files:
        img_id = file.replace(".nii.gz", "")
        with tempfile.TemporaryDirectory() as tempdir1:
            # transfer labels to tempdir
            shutil.copy(f'./step1data/{file}', tempdir1)
            # label map to generate images from
            path_label_map = tempdir1

            # general parameters
            n_examples = 5
            result_dir = args.out_dir
            output_shape = None  # shape of the output images, obtained by randomly cropping the generated images

            # specify structures from which we want to generate
            generation_labels = f'./step2data/generation_labels_{img_id}.npy'
            # specify structures that we want to keep in the output label maps
            output_labels = f'./step2data/output_labels_{img_id}.npy'
            # we regroup structures into K classes, so that they share the same distribution for image generation
            generation_classes = f'./step2data/generation_classes_{img_id}.npy'

            # We specify here that we type of prior distributions to sample the GMM parameters.
            # By default prior_distribution is set to 'uniform', and in this example we want to change it to 'normal'.
            prior_distribution = 'normal'
            # We specify here the hyperparameters of the prior distributions to sample the means of the GMM.
            # As these prior distributions are Gaussians, they are each controlled by a mean and a standard deviation.
            # Therefore, the numpy array pointed by prior_means is of size (2, K), where K is the nummber of classes specified in
            # generation_classes. The first row of prior_means correspond to the means of the Gaussian priors, and the second row
            # correspond to standard deviations.
            prior_means = f'./step3data/prior_means_{img_id}.npy'
            # same as for prior_means, but for the standard deviations of the GMM.
            prior_stds = f'./step3data/prior_stds_{img_id}.npy'

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
                                os.path.join(result_dir, f'images/{img_id}_%s.nii.gz' % n))
                utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                                os.path.join(result_dir, f'labels/{img_id}_%s.nii.gz' % n))
        

    