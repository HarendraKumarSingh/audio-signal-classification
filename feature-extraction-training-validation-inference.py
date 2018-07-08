
# coding: utf-8

# When you load the data, it gives you two objects: a numpy array of an audio file and the corresponding 
# sampling rate by which it was extracted. Now to represent this as a waveform

#get_ipython().magic(u'pylab inline')
import os
import pandas as pd
import librosa
import glob 
import matplotlib
import librosa.display

import csv
import numpy as np

#Read csv
train = pd.read_csv('./data/train/train.csv')


data_dir = './data/train/'
def parser(row):
   # function to load files and extract features
   file_name = os.path.join(os.path.abspath(data_dir), 'Train', str(row.ID) + '.wav')

   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      print X, sample_rate

      # we extract mfcc feature from data
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
   except Exception as e:
      print("Error encountered while parsing file: ", file)
      return None, None
 
   feature = mfccs
   label = row.Class
 
   return [feature, label]


# Convert the data to pass it in our deep learning model

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import tensorflow

temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']

X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())

lb = LabelEncoder()

y = to_categorical(lb.fit_transform(y))


# Testing/Validation
# Creating validation dataset for our deep learning model
test = pd.read_csv('./data/test/test.csv')

data_dir = './data/test/'
def parser2(row):
    # function to load files and extract features
    file_name = os.path.join(os.path.abspath(data_dir), 'Test', str(row.ID) + '.wav')

    # handle exception to check if there isn't a file which is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        print X, sample_rate
        
        # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None
 
    feature = mfccs
    label = row.Class
    return [feature, label]


# Convert the validation data to pass it in our deep learning model

test_temp = test.apply(parser2, axis=1)
test_temp.columns = ['feature', 'label']


val_X = np.array(test_temp.feature.tolist())
val_y = np.array(test_temp.label.tolist())


lb2 = LabelEncoder()

val_y = to_categorical(lb2.fit_transform(val_y))


# Run a deep learning model and get results

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
# from keras.utils import np_utils
from sklearn import metrics 


num_labels = y.shape[1]
filter_size = 2

# build model
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


#Model training and saving the model file (architecture + weights + optimizer state)

model.fit(X, y, batch_size=32, nb_epoch=117, validation_data=(val_X, val_y))

model.save('audio_model.h5')  # creates a HDF5 file 'audio_model.h5'
print "Model file saved."

# evaluate loaded model on test data
loaded_model = load_model('audio_model.h5')

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print "Model loaded from disk"

score = loaded_model.evaluate(val_X, val_y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# Inference part
preds = loaded_model.predict(val_X)
print "Predicted Classes : ", preds

df = pd.DataFrame(preds)
df.to_csv('predict_output.csv', index=False, encoding='utf-8')

