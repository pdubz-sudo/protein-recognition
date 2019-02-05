import os
import numpy as np
from PIL import Image

resize_shape = 224,224
original_images_path = r"D:\kaggle-kaggle-protien-data\scripts\practice-images"
save_path_dir = r"D:\kaggle-kaggle-protien-data\scripts\practice-images-small"


for subdir, dirs, files in os.walk(original_images_path):
	for file in files:
		filepath = subdir + os.sep + file
		small_file_name = os.path.splitext(file)[0] + '_small'
		img = Image.open(filepath)

		assert np.asarray(img).shape == (512,512), 'An image is not the right shape (512, 512)'
		# default is bicubic for pillow 2.7 and higher. https://pillow.readthedocs.io/en/3.0.x/releasenotes/2.7.0.html#default-filter-for-thumbnails
		resized_img = img.resize(resize_shape)
		resized_img.save(save_path_dir + os.sep + small_file_name + '.png')
		
		del resized_img, img