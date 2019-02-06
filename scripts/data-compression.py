from __future__ import print_function

import os
from PIL import Image
import numpy as np
import h5py

# This script compresses all the original images into resized arrays for
# train images and test images in an hdf5 file. Since the images will be converted to arrays, 
# their respective ids have also been stored as arrays for identification.
# The directory of the train and test images must be indicated. The directory
# for where the hdf5 file will be saved must also be indicated.
# In addition, the resize shape size can be changed.

# The data can be retrieved from the .h5 file the code below.
# with h5py.File(compressed_data_dir + os.sep + 'data.h5', 'r') as hf:
#     train = hf['train'][:]
#     train_ids = hf['train_ids'][:]
#     test = hf['test'][:]
#     test_ids = hf['test_ids'][:]

print('\nThis will take a while...')

traindata_path = r"D:\kaggle-kaggle-protien-data\train"
testdata_path = r"D:\kaggle-kaggle-protien-data\test"

compressed_data_dir = r"D:\kaggle-kaggle-protien-data\data"

resize_shape = 224,224



image_arrays = []
ids = []

for subdir, dirs, files in os.walk(traindata_path):
    for file in files:
        # print(subdir + os.sep + file)
        filepath = subdir + os.sep + file
        # small_file_name = os.path.splitext(file)[0] + '_small'
        img = Image.open(filepath)
        
        assert np.asarray(img).shape == (512,512), 'An image is not the right shape (512, 512)'
        # default is bicubic for pillow 2.7 and higher. https://pillow.readthedocs.io/en/3.0.x/releasenotes/2.7.0.html#default-filter-for-thumbnails
        resized_img = img.resize(resize_shape) 
        imgarr = np.asarray(resized_img).flatten()
        image_arrays.append(imgarr)
        
        
        imageId = file.split('_')[0]
        if imageId not in ids:
            ids.append(imageId)
        
        
        del img, resized_img, imgarr, filepath, imageId

    
train = np.reshape(image_arrays, newshape=(-1, 4, resize_shape[0], resize_shape[1]))
ids = np.asarray(ids, dtype='S')

with h5py.File(compressed_data_dir + os.sep + 'data.h5', 'w') as hf:
    hf.create_dataset('train', data=train)
    hf.create_dataset('train_ids', data=ids)
    
del image_arrays, ids, train


image_arrays = []
ids = []
for subdir, dirs, files in os.walk(testdata_path):
    for file in files:
        # print(subdir + os.sep + file)
        filepath = subdir + os.sep + file
        # small_file_name = os.path.splitext(file)[0] + '_small'
        img = Image.open(filepath)
        
        assert np.asarray(img).shape == (512,512), 'An image is not the right shape (512, 512)'
        # default is bicubic for pillow 2.7 and higher. https://pillow.readthedocs.io/en/3.0.x/releasenotes/2.7.0.html#default-filter-for-thumbnails
        resized_img = img.resize(resize_shape) 
        imgarr = np.asarray(resized_img).flatten()
        image_arrays.append(imgarr)
        
        
        imageId = file.split('_')[0]
        if imageId not in ids:
            ids.append(imageId)
        
        
        del img, resized_img, imgarr, filepath, imageId

test = np.reshape(image_arrays, newshape=(-1, 4, resize_shape[0], resize_shape[1]))
ids = np.asarray(ids, dtype='S')

with h5py.File(compressed_data_dir + os.sep + 'data.h5', 'a') as hf:
    hf.create_dataset('test', data=test)
    hf.create_dataset('test_ids', data=ids)
    
del image_arrays, ids, test