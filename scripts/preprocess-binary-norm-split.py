from __future__ import print_function
from __future__ import division

import h5py
from PIL import Image
import numpy as np
import pandas as pd

'''This script reads in the data file of image arrays and ids that are in uint8 data type
and preps the data for model training. It creates binary labels, 0 and 1, for any selected organelle which 
is indicated by its key for 'key target'. The data is normalized (image normalization) to a np.float32
datatype and then split into train, val, and test sets. Their respective labels are also split.
The reason that this script is seperate from the source data is the np.float32 make the dataset
almost 3 times bigger than the uint8 datatype. So, this script simplifies making new binary target 
training data.'''


# Path to dataset of image array that have datatype uint8 and S bytes.
dataset_path = r'\data\data.h5'
label_path = r'\data\train_labels.csv'

file_save_path = r'\data\target0-norm-split.hdf5'

### Get labels with organelle of interest
# 0: 'Nucleoplasm',  
# 1: 'Nuclear membrane',   
# 2: 'Nucleoli',   
# 3: 'Nucleoli fibrillar center',   
# 4: 'Nuclear speckles',   
# 5: 'Nuclear bodies',   
# 6: 'Endoplasmic reticulum',   
# 7: 'Golgi apparatus',   
# 8: 'Peroxisomes',   
# 9: 'Endosomes',   
# 10: 'Lysosomes',   
# 11: 'Intermediate filaments',   
# 12: 'Actin filaments',   
# 13: 'Focal adhesion sites',   
# 14: 'Microtubules',   
# 15: 'Microtubule ends',   
# 16: 'Cytokinetic bridge',   
# 17: 'Mitotic spindle',   
# 18: 'Microtubule organizing center',   
# 19: 'Centrosome',   
# 20: 'Lipid droplets',   
# 21: 'Plasma membrane',   
# 22: 'Cell junctions',   
# 23: 'Mitochondria',   
# 24: 'Aggresome',   
# 25: 'Cytosol',   
# 26: 'Cytoplasmic bodies',   
# 27: 'Rods & rings' 

# Select a target
targetId = 0





print('\nThis will take a while...')
# Ingest data
# %%time

with h5py.File(dataset_path, 'r') as f:
    train = f['train'][:]#.astype(np.float32)
    trainIds = f['train_ids'][:]
    
# The current dype is uint8. Need to change dtype.
train = train.astype(np.float32)

# Change channel position to make it easier for pre-trained model.
train = np.moveaxis(train, 1,3)
print('Training set shape: ',train.shape)

# Normalize image arrays
# %%time

train = np.divide(train, 255)


### Make Y labels for the model
dfLabels = pd.read_csv(label_path)
dfLabels['Target'] = dfLabels['Target'].apply(lambda x: x.split())
idx = [i for i, x in enumerate(dfLabels['Target']) for organelle in x if organelle == str(targetId)]
print('Number of samples for targetId {}: {}'.format(targetId, len(idx)))

# Get image id's for for samples that have that organelle.
labelIds = dfLabels.iloc[idx,:]['Id']

# Convert to bytes for matching in the hdf5 label_ids array
labelIds = [str.encode(Id) for Id in labelIds]

# Fastest and most efficient way to getting matches of 2 lists.
# Take the index which will be used to find training examples with a label.
# https://stackoverflow.com/questions/29452735/find-the-indices-at-which-any-element-of-one-list-occurs-in-another
st = set(labelIds)
positiveLabel = [i for i, x in enumerate(trainIds) if x in st]

# Initizlize Y
Y = np.zeros((len(trainIds), 1))
# Fill positive label with positive index list
Y[positiveLabel,:] = 1

def train_val_test_split(Xdata, Ydata, train_size, val_size):
    '''Shuffle and split data into 3 sub-sets: train, validation, and test.
    
    Arguments:
    Xdata -- numpy array, training data. m must be (m, ...)
    Ydata -- numpy array, label data. m must be (m, ...)
    train_size -- float, split ratio. Ex. 0.8 for 80% for train data.
    val_size -- float, split ratio of desired amount. Ex. 0.2 for 20% for val data.
    
    Return:
    Xdata_train -- X train set
    Xdata_val -- X val set
    Xdata_test -- X test set
    
    Ydata_train -- Y train set
    Ydata_val -- Y val set
    Ydata_test -- Y test set
    '''
    
    assert Xdata.shape[0] == Ydata.shape[0], 'Train and Label examples are not equal.'
    
    shuff_id = np.random.permutation(train.shape[0])
    
    train_count = np.floor(Xdata.shape[0]*train_size).astype(int)
    val_count = np.floor(Xdata.shape[0]*val_size).astype(int)
    
    train_ix, val_ix, test_ix = np.split(shuff_id, [train_count, train_count+val_count])
    
    Xdata_train = Xdata[train_ix, ...]
    Xdata_val = Xdata[val_ix, ...]
    Xdata_test = Xdata[test_ix, ...]
        
    Ydata_train = Ydata[train_ix, ...]
    Ydata_val = Ydata[val_ix, ...]
    Ydata_test = Ydata[test_ix, ...]
        
    return Xdata_train, Xdata_val, Xdata_test, Ydata_train, Ydata_val, Ydata_test

X_train, X_val, X_test, Y_train, Y_val, Y_test = train_val_test_split(train, Y, 0.7, 0.15)

assert X_train.shape[0]+X_val.shape[0]+X_test.shape[0]==train.shape[0]

del train, Y

with h5py.File(file_save_path, 'w') as hf:
	hf.create_dataset('X_train', data=X_train)
	hf.create_dataset('X_val', data=X_val)
	hf.create_dataset('X_test', data=X_test)

	hf.create_dataset('Y_train', data=Y_train)
	hf.create_dataset('Y_val', data=Y_val)
	hf.create_dataset('Y_test', data=Y_test)