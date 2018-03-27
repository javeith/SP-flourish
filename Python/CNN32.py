
# coding: utf-8

# In[ ]:

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# Turn inFile into a one hot label
def createOneHot(inFile,amountOfClasses):
    outFile = np.zeros((inFile.shape[0],inFile.shape[1], amountOfClasses));
    for i in range(0,inFile.shape[0]):
        for j in range(0,inFile.shape[1]):
            #if inFile[i,j] != 0:
            outFile[i,j,int(inFile[i,j])] = 1;
    return outFile;

# Turn 41 orthomosaics into one array. From 1 path with Bands 1 to 41
def loadOneOrthomosaic(imagePath):
    temp= np.asarray( Image.open( imagePath + 'Band1.png') );
    inputData = np.zeros(( temp.shape[0], temp.shape[1], inputChannels),dtype=np.dtype('uint8'))
    inputData[:,:,0] = temp[:,:]
    for i in range(1,41):
        temp= np.asarray( Image.open( imagePath + 'Band' + str(i+1) + '.png') );
        inputData[:,:,i] = temp[:,:];
    return np.expand_dims( inputData, axis=0)

# Load one label from a .mat
def loadOneLabel(imagePath):
    temp = sio.loadmat(imagePath);
    trainOneHotLabels =  createOneHot(temp['labeledPicture'], amountOfClasses);
    return np.expand_dims( trainOneHotLabels, axis=0)

# Read training data and labels for an array of path tuples, returns (data,labels)
def readSet(paths):
    numberOfImages = len(paths);
    data = np.zeros( (numberOfImages,imageSize[1],imageSize[0],inputChannels), dtype=np.dtype('uint8'))
    labels = np.zeros((numberOfImages,imageSize[1],imageSize[0],amountOfClasses), dtype=np.dtype('uint8'))
    for i in range(0,numberOfImages):
        data[i,:,:,:] = loadOneOrthomosaic(paths[i][0])
        print('Loaded '+paths[i][0]+'.')
        labels[i,:,:,:] = loadOneLabel(paths[i][1])
        print('Loaded '+paths[i][1]+'.')
    return data,labels

def oneHotToBinary(oneHotLabel,threshold):
    temp = np.zeros((oneHotLabel.shape[1],oneHotLabel.shape[2]), dtype=np.dtype('uint8') );
    for i in range(0,oneHotLabel.shape[1]):
        for j in range(0,oneHotLabel.shape[2]):
            #temp[i,j] = np.argmax(oneHotLabel[0,i,j,:]) + 1;
            if np.argmax(oneHotLabel[0,i,j,:])>threshold:
                temp[i,j] = np.argmax(oneHotLabel[0,i,j,:])
    return temp

def predictAndSaveOneImage(inPath,outPath,FCNmodel,threshold):
    testImage = loadOneOrthomosaic(inPath);
    testLabelPredict = FCNmodel.predict(testImage);
    testLabelPredictBinary = oneHotToBinary(testLabelPredict,threshold);
    sio.savemat(outPath, mdict={'testLabelPredict': testLabelPredictBinary});

def predictAndSaveSet(pathArray, FCNmodel,threshold):
    numberOfImages = len(pathArray);
    for i in range(0,numberOfImages):
        predictAndSaveOneImage(pathArray[i][0], pathArray[i][1], FCNmodel,threshold)
        print('Saved ' + pathArray[i][1])
        
def plotImage(image):
    plt.imshow(image, cmap='gray')
    plt.show()
    
def getTrainPaths(paths):
    folders = ['FIP','Ximea_Tamron']
    fullPath = []
    for imageType in paths:
        for date in imageType:
            temp = []
            temp.append(basePath+'trainData/'+folders[paths.index(imageType)]+'/'+date+'/')
            temp.append(basePath+'trainLabels/'+folders[paths.index(imageType)]+'_'+date+'.mat') 
            fullPath.append(temp)
    return fullPath
    
def getTestPaths(paths):
    folders = ['FIP','Ximea_Tamron']
    fullPath = []
    for imageType in paths:
        for date in imageType:
            temp = []
            temp.append(basePath+'testData/'+folders[paths.index(imageType)]+'/'+date+'/')
            temp.append(basePath+'testLabelPredict/'+folders[paths.index(imageType)]+'_'+date+'.mat') 
            fullPath.append(temp)
    return fullPath

def getSampleWeight(label):
    counter = np.zeros(amountOfClasses)
    numberOfElements = label.shape[0]*label.shape[1]
    for i in range(0,label.shape[1]):
        for j in range(0,label.shape[2]):
            for k in range(0,label.shape[0]):
                counter[ np.argmax(label[k,i,j]) ] += 1
    counter = sum(counter)/counter
    print('Class weights:')
    print(counter)
    return counter
    



"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
from keras import backend as K
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

# Data paths
basePath = '/cluster/home/jveith/LeonhardData/';

trainSets = [
    #FIP
    [
        '20170622'
    ],
    #Ximea_Tamron
    [
        '20170510',
        '20170622'
    ]
]

validationSets = [
    #FIP
    [
        '20170531'
    ],
    #Ximea_Tamron
    [
    ]
]

testSets = [
    #FIP
    [
        '20170531', 
        '20170622',
        '20170531_cropped',
        '20170622_cropped',
        '20170802'
    ],
    #Ximea_Tamron
    [
        '20170510',
        '20170622',
        '20170510_cropped',
        '20170622_cropped',
        '20170613'
    ]
]
    
trainPaths = getTrainPaths(trainSets)
validationPaths = getTrainPaths(validationSets)
testPaths = getTestPaths(testSets)

amountOfClasses = 9;
inputChannels = 41;
imageSize = [1600,1600]; # [X,Y]

trainData,trainLabels = readSet(trainPaths)
validationData,validationLabels = readSet(validationPaths)

from keras.models import *
from keras.layers import Conv2D, MaxPooling2D, Input, Cropping2D, UpSampling2D,concatenate,Dropout

model_inputs = Input(shape=(imageSize[0], imageSize[1], inputChannels))

conv1 = Conv2D(16,3, activation='relu', padding='same')(model_inputs)
conv1 = Conv2D(16,3, activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

conv2 = Conv2D(32,3, activation='relu', padding='same')(pool1)
conv2 = Conv2D(32,3, activation='relu', padding='same')(conv2)
drop2 = Dropout(0.5)(conv2)

upconv3 = Conv2D(16,2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop2))
concat3 = concatenate([upconv3,conv1],axis=3)
conv3 = Conv2D(16,3, activation='relu', padding='same')(concat3)
conv3 = Conv2D(16,3, activation='relu', padding='same')(conv3)

conv4 = Conv2D(amountOfClasses, (1,1), activation='softmax', padding='valid')(conv3)

CNNModel = Model(inputs = model_inputs, outputs = conv4)
print(CNNModel.summary())

#weights = getSampleWeight(trainLabels)
weights = np.array([.1,1,1,1,1,1,1,1,1])
wLoss = weighted_categorical_crossentropy(weights)
optAdam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
CNNModel.compile(loss= wLoss, optimizer= optAdam,  metrics=['accuracy'])

CNNModel.fit(trainData, trainLabels,validation_data=(validationData,validationLabels), epochs=50)

predictAndSaveSet(testPaths, CNNModel,.1)

