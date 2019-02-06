# Image Recognition

# Currently working on this for fun...

Image recognition of cell organelles with deep learning using TensorFlow.


Data can be downloaded here
https://www.kaggle.com/c/human-protein-atlas-image-classification

Completed so far:
- Some EDA which can be found in the jupyter notebook.
- Script that reduces images size.
- Pre-processing: Script that reduces images size, converts to array (m, channel,h, w), saves as hdf5:
	1. download test and train images
	2. Make seperate folder for train and test images
	3. Use the data-compression.py script and specify the path to the train and test images, hdf5 file output 
	directory, and image size. The original files MUST be 512 x 512. 