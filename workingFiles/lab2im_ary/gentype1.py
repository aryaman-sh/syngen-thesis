import os
from lab2im import utils
from lab2im.image_generator import ImageGenerator

exp_nums = [1,2,3,4,5,6,7,8,9]

result_dir = '/scratch/itee/uqasha24/anshuexp'
utils.mkdir(result_dir)

n = 10

for exp in exp_nums:
	path_label_map = f'organ/{str(exp)}.nii.gz'
	generation_labels = f'organ/{str(exp)}.npy'
	brain_generator = ImageGenerator(path_label_map, generation_labels=generation_labels)

	for n in range(1, n+1):	
		im, lab = brain_generator.generate_image()
		utils.save_volume(im, brain_generator.aff, brain_generator.header, os.path.join(result_dir, f'scan_{str(exp)}_{str(n)}.nii.gz'))
		utils.save_volume(lab, brain_generator.aff, brain_generator.header, os.path.join(result_dir, f'labels_{str(exp)}_{str(n)}.nii.gz'))

