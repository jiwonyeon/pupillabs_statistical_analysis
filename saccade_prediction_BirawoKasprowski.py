from scipy.io import loadmat
import pandas as pd
import numpy as np
import os, glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPool1D, Conv2D, BatchNormalization
import netron
import seaborn as sns


"""
    This code is from the article by Birawo and Kasprowski (2022). 
    https://github.com/mebirtukan/Evaluation-of-eye-movement-event-detection-algorithms.git    

    It is an example of classfication of saccades using CNN.
    Jiwon Yeon. 2024.
"""

def openfile(filename):
    mat = loadmat(filename)
    mdata = mat['ETdata']
    mtype = mdata.dtype
    ndata = {n: mdata[n][0,0] for n in mtype.names}
    data_headline = ndata['pos']
    data_headline = data_headline[0]
    data_raw = ndata['pos']
    pdata = pd.DataFrame(data_raw,columns=data_headline)
    df=pd.DataFrame(pdata)
    df[1.0]=df[1.0].astype(int)
    #t=pdata.iloc[:, 0].values maybe we should use it...
    x=pdata.iloc[:, 3:5].values
    y=pdata.iloc[:, 5].values
    print("File",filename,"opened")
    return x ,y
     
# # data has two coordinates: X,Y returns velX and velY
# def calc_xy_velocity(data):
#     velX = [] #x values difference
#     velY = [] #y values difference 

#     for i in range(len(data) - 1):
#         velX.append(float(data[i+1,0]) - float(data[i,0]) ) # 2ms!
#         velY.append(float(data[i+1,1]) - float(data[i,1]) )
#     velX = np.array(velX)
#     velY = np.array(velY)
#     velocity = np.vstack([velX,velY]).T
#     return velocity

# # data has two coordinates: X,Y returns ONE velocity
# def calc_velocity(data):
#     velX = [] #x values difference
#     velY = [] #y values difference 
#     for i in range(len(data) - 1):
#         velX.append(float(data[i+1,0]) - float(data[i,0]) ) # 2ms!
#         velY.append(float(data[i+1,1]) - float(data[i,1]) )
#     velX = np.array(velX)
#     velY = np.array(velY)
#     velocity = np.sqrt(np.power(velX,2) + np.power(velY,2))
#     print(velocity.shape)  
#     return velocity

def open_list_of_files(files_to_load):
    samples = []
    labels =[]
    for my_file in files_to_load:
        sam,lab = openfile(my_file)
        #ssam,slab = make_sequences(sam,lab,50)
        print('Number of samples so far:',len(samples))
        samples.extend(sam)
        labels.extend(lab)
    samples = np.array(samples)
    labels = np.array(labels)
    print('Number of samples at the end:',len(samples))
    return samples,labels

# def make_sequences(samples, labels, sequence_dim = 100, sequence_lag = 1, sequence_attributes = 2):
#     nsamples = []
#     nlabels = [] 
#     for i in range(0,samples.shape[0]-sequence_dim,sequence_lag):
#             nsample = np.zeros((sequence_dim,sequence_attributes))
#             for j in range(i,i+sequence_dim):
#                 nsample[j-i,0] = samples[j,0]
#                 nsample[j-i,1] = samples[j,1]
#             nlabel = labels[i+sequence_dim//2]
#             #print("Sample",nsample)
#             #print("Label",nlabel)
#             nsamples.append(nsample)
#             nlabels.append(nlabel)
        
#     samples = np.array(nsamples)
#     labels = np.array(nlabels)
#     return samples,labels 

# Download the sample data if it hasn't
if not os.path.exists('BirawoKasprowski'):
    import subprocess
    os.makedirs('BirawoKasprowski')  # Create the directory
    os.chdir('BirawoKasprowski')     # Change to the directory    
    # Download the file using curl
    subprocess.run(["curl", "-O", "http://www.kasprowski.pl/datasets/events.zip"])
    # Unzip the file
    subprocess.run(["unzip", "events.zip"])
    os.chdir('..')                   # Optionally, return to the original directory

files_to_load = ['BirawoKasprowski/UH33_img_vy_labelled_MN.mat']
# files_to_load1 = ['data/TH34_img_Europe_labelled_MN.mat', 'data/UH21_img_Rome_labelled_RA.mat']

s,l = open_list_of_files(files_to_load)
print(s.shape)
# s1,l1=open_list_of_files(files_to_load1)
# print(s.shape)
# print(l.shape)
# # print(s1.shape)
# # print(l1.shape)
     
# s = calc_xy_velocity(s)
# s1 = calc_xy_velocity(s1)

# sequence_dim = 100
# print("Samples shape before sequencing",s.shape)

# print("Converting to sequences of length {}".format(sequence_dim))
# x, y = make_sequences(s, l, sequence_dim)
# x1, y1 = make_sequences(s1, l1, sequence_dim)
# print("Samples shape after sequencing: {}".format(x.shape))
# print("Labels shape after sequencing: {}".format(y.shape))     

# lb = LabelBinarizer()
# lb.fit(y)
# y = lb.transform(y)
# y1 = lb.transform(y1)


# #%% Build CNN Model

# inputShape = (sequence_dim, 2)
# #inputShape = (x.shape)
# print('inputShape:',inputShape)
# model = Sequential()
# model.add(Conv1D(32, 3,input_shape=inputShape))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(Dropout(0.2))

# model.add(Conv1D(64, 3, padding="same"))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(Dropout(0.2))
# model.add(Conv1D(128, 3, padding="same"))
# model.add(Activation("relu"))
# model.add(Dropout(0.2))
# model.add(Flatten())
# #model.add(Dense(128, activation='sigmoid'))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(3, activation='softmax'))
# #model.add(Dense(3, activation='softmax'))

# model.summary()
# model.save('CNN_CLASS.h5')

# #%% training and testing
# model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])
    
# EPOCHS=20
# BATCH=100
# model.fit(x, y, batch_size=BATCH, epochs=EPOCHS
#               ,validation_data=(x1,y1)
#               )

# print("Training")
# cnnResults = model.predict(x)
# print(confusion_matrix(y.argmax(axis=1), cnnResults.argmax(axis=1)))
# print(classification_report(y.argmax(axis=1), cnnResults.argmax(axis=1)))
# print("CNN Accuracy: {:.2f}".format(accuracy_score(y.argmax(axis=1), cnnResults.argmax(axis=1))))
# print("Cohen's Kappa {:.2f}".format(cohen_kappa_score(y.argmax(axis=1), cnnResults.argmax(axis=1))))

# print("Test")
# cnnResults = model.predict(x1)
# print(confusion_matrix(y1.argmax(axis=1), cnnResults.argmax(axis=1)))
# print(classification_report(y1.argmax(axis=1), cnnResults.argmax(axis=1)))
# CM=(confusion_matrix(y1.argmax(axis=1), cnnResults.argmax(axis=1)))
# print("CNN Accuracy: {:.2f}".format(accuracy_score(y1.argmax(axis=1), cnnResults.argmax(axis=1))))
# print("Cohen's Kappa {:.2f}".format(cohen_kappa_score(y1.argmax(axis=1), cnnResults.argmax(axis=1))))

# #%% print out the testing result
# cm_normalized=np.round(CM/np.sum(CM, axis=1).reshape(-1, 1), 2)
# print(cm_normalized)
# sns.heatmap(cm_normalized, cmap='Blues', annot=True, cbar_kws={"orientation": "vertical", "label": "color bar"}, xticklabels=['fix','sac','pso'], yticklabels=['fix', 'sac', 'pso'])
# plt.xlabel("Predicted value")
# plt.ylabel("Actual value")
# plt.show()