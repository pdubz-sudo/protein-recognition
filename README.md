# Image Recognition

# Currently working on this for fun...

Image recognition of cell organelles with deep learning using TensorFlow.


Data can be downloaded here
https://www.kaggle.com/c/human-protein-atlas-image-classification

Completed so far:
- [x] Some EDA which can be found in the jupyter notebook.
- [x] Script that reduces images size.
- [x] Pre-processing: Script that reduces images size, converts images to arrays (m, channel,h, w), and saves in a HDF5 file. If you're
unfamiliar with HDF5 files, you can read about them here https://www.h5py.org/. To run the pre-process script, do the following:
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

- [x] Further Pre-processing: Script that creates and saves binary normalized train, val, and test data for training CNN models.
The created file is 24GB.
