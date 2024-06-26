"""
This tutorials generates 5 synthetic *T1-weighted* brain MRI scans from a label map.
Specifically, it explains how to impose prior distributions on the GMM parameters, so that we can can generate images
of desired intensity distribution.
By default the GMM parameters (means and standard deviations of each Gaussian), are sampled from uniform distributions
of wide predefined ranges, thus yielding output images of random contrast.
Here we show how to generate images of desired contrast by imposing the prior distributions from which we sample the
means and standard deviations of the GMM.


If you use this code, please cite the first SynthSeg paper:
https://github.com/BBillot/lab2im/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

#temp fix somehow fixes path stuff
# run this from inside the tutorials directory
import sys
sys.path.append('../')

import os
import time
from lab2im import utils
from lab2im.image_generator import ImageGenerator


# label map to generate images from
path_label_map = './data_example/brain_label_map.nii.gz'

# general parameters
n_examples = 5
result_dir = './generated_images'
output_shape = None  # shape of the output images, obtained by randomly cropping the generated images

# specify structures from which we want to generate
generation_labels = './data_example/generation_labels.npy'
# specify structures that we want to keep in the output label maps
output_labels = './data_example/segmentation_labels.npy'
# we regroup structures into K classes, so that they share the same distribution for image generation
generation_classes = './data_example/generation_classes.npy'

# We specify here that we type of prior distributions to sample the GMM parameters.
# By default prior_distribution is set to 'uniform', and in this example we want to change it to 'normal'.
prior_distribution = 'normal'
# We specify here the hyperparameters of the prior distributions to sample the means of the GMM.
# As these prior distributions are Gaussians, they are each controlled by a mean and a standard deviation.
# Therefore, the numpy array pointed by prior_means is of size (2, K), where K is the nummber of classes specified in
# generation_classes. The first row of prior_means correspond to the means of the Gaussian priors, and the second row
# correspond to standard deviations.
prior_means = './data_example/prior_means.npy'
# same as for prior_means, but for the standard deviations of the GMM.
prior_stds = './data_example/prior_stds.npy'

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
                      os.path.join(result_dir, 't1_%s.nii.gz' % n))
    utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 't1_labels_%s.nii.gz' % n))
