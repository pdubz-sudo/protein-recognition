# Scripts for preparing data for training CNN model

Data can be downloaded here
https://www.kaggle.com/c/human-protein-atlas-image-classification

### Resize_images.py
Resizes the images to a new directory. The images must be 512 x 512 so the script works for the smaller download set, which is still pretty large.
If you want to use this script with 250GB dataset then you must remove the assert statement in the script.

### data-compression.py 
Script that reduces images size, converts images to arrays (m, channel,h, w), and saves in a 
HDF5 file. If you're unfamiliar with HDF5 files, you can read about them here https://www.h5py.org/. To run the pre-process 
script, do the following:
1. Download test and train images (NOT the ~250GB version)
2. Make seperate folders for train and test images.
3. Use the data-compression.py script and specify the path to the train and test images, hdf5 file output 
directory, and image size. The original files MUST be 512 x 512. The pre-proceed HDF5 file will be 8GB
and can be accessed by the following code:

        with h5py.File(compressed_data_dir + os.sep + 'data.h5', 'r') as hf:
            train = hf['train'][:]
            train_ids = hf['train_ids'][:]
            test = hf['test'][:]
            test_ids = hf['test_ids'][:]

### preprocess-binary-norm-split.py
Script that creates and saves binary normalized train, val, and test data for training CNN models.
The created file is 24GB.

		with h5py.File('./data/target0-norm-split.hdf5', 'r') as hf:
    		X_train = hf['X_train'][:]
    		X_val = hf['X_val'][:]
    		X_test = hf['X_test'][:]
    		Y_train = hf['Y_train'][:]
    		Y_val = hf['Y_val'][:]
    		Y_test = hf['Y_test'][:]