
# coding: utf-8

# In[5]:

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
import imageio
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
            outFile[i,j,int(inFile[i,j]-1)] = 1;
    return outFile;

# Turn 41 orthomosaics into one array. From 2 separate paths, 1-25 and 1-16
def loadOneOrthomosaic(imagePath1,imagePath2):
    temp= imageio.imread(imagePath1 + '1.png');
    inputData = np.zeros( (temp.shape[0],temp.shape[1],inputChannels) ,dtype=np.dtype('uint8'))
    for i in range(1,25):
        temp= imageio.imread(imagePath1 + 'Band' + str(i) + '.png');
        inputData[:,:,i] = temp[:,:,0];
    for i in range(1,16):
        temp= imageio.imread(imagePath2 + 'Band' + str(i) + '.png');
        inputData[:,:,i+25] = temp[:,:,0];
    return np.expand_dims(inputData,axis=0)

# Turn 41 orthomosaics into one array. From 1 path with Bands 1 to 41
def loadOneOrthomosaic(imagePath):
    temp= imageio.imread(imagePath + 'Band1.png');
    inputData = np.zeros(( temp.shape[0], temp.shape[1], inputChannels),dtype=np.dtype('uint8'))
    for i in range(1,41):
        temp= imageio.imread(imagePath + 'Band' + str(i) + '.png');
        inputData[:,:,i] = temp[:,:];
    return np.expand_dims( inputData, axis=0)

# Load one label from a .mat
def loadOneLabel(imagePath):
    temp = sio.loadmat(imagePath);
    trainOneHotLabels =  createOneHot( temp['labeledPicture'], amountOfClasses);
    return np.expand_dims( trainOneHotLabels, axis=0)

# Read training data and labels for an array of path tuples, returns (data,labels)
def readSet(paths):
    numberOfImages = len(paths);
    data = np.zeros( (numberOfImages,imageSize[1],imageSize[0],inputChannels), dtype=np.dtype('uint8'));
    labels = np.zeros((numberOfImages,imageSize[1],imageSize[0],amountOfClasses), dtype=np.dtype('uint8'))
    for i in range(0,numberOfImages):
        data[i,:,:,:] = loadOneOrthomosaic(paths[i][0]);
        labels[i,:,:,:] = loadOneLabel(paths[i][1]);
    return data,labels

def oneHotToCategorical(oneHotLabel):
    temp = np.zeros((oneHotLabel.shape[1],oneHotLabel.shape[2]), dtype=np.dtype('uint8') );
    for i in range(0,oneHotLabel.shape[1]):
        for j in range(0,oneHotLabel.shape[2]):
            #temp[i,j] = np.argmax(oneHotLabel[0,i,j,:]) + 1;
            temp[i,j] = np.argmax(oneHotLabel[0,i,j,:]);
    return temp

def predictAndSaveOneImage(inPath,outPath,FCNmodel):
    testImage = loadOneOrthomosaic(inPath);
    testLabelPredict = FCNmodel.predict(testImage);
    testLabelPredictCat = oneHotToCategorical(testLabelPredict);
    sio.savemat(outPath, mdict={'testLabelPredict': testLabelPredictCat});

def predictAndSaveSet(pathArray, FCNmodel):
    numberOfImages = len(pathArray);
    for i in range(0,numberOfImages):
        predictAndSaveOneImage(pathArray[i][0], pathArray[i][1], FCNmodel)
        print('Saved ' + pathArray[i][1])
        
def plotImage(image):
    plt.imshow(image)
    
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


# In[9]:

# Data paths
basePath = '/media/hdd/jannicveith/xFcnClassifier/';

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

# Data parameters
amountOfClasses = 9;
inputChannels = 41;
imageSize = [1600,1600]; # [X,Y]

numberOfEpochs = 20;


# In[3]:

from keras.models import *
from keras.layers import Conv2D, MaxPooling2D, Input, Cropping2D, UpSampling2D,concatenate,Dropout

#crop1 = Cropping2D(cropping=((0, 0), (0, 0)))(conv1) # cropping=((Top, Bottom), (Left, Right))

model_inputs = Input(shape=(imageSize[0], imageSize[1], inputChannels))

conv1 = Conv2D(64,3, activation='relu', padding='same')(model_inputs)
conv1 = Conv2D(64,3, activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

conv2 = Conv2D(128,3, activation='relu', padding='same')(pool1)
conv2 = Conv2D(128,3, activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

conv3 = Conv2D(256,3, activation='relu', padding='same')(pool2)
conv3 = Conv2D(256,3, activation='relu', padding='same')(conv3)
drop3 = Dropout(0.5)(conv3)

upconv4 = Conv2D(128,2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop3))
concat4 = concatenate([upconv4,conv2],axis=3)
conv4 = Conv2D(128,3, activation='relu', padding='same')(concat4)
conv4 = Conv2D(128,3, activation='relu', padding='same')(conv4)

upconv5 = Conv2D(64,2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv4))
concat5 = concatenate([upconv5,conv1],axis=3)
conv5 = Conv2D(64,3, activation='relu', padding='same')(concat5)
conv5 = Conv2D(64,3, activation='relu', padding='same')(conv5)

conv6 = Conv2D(amountOfClasses, (1,1), activation='softmax', padding='valid')(conv5)


UNetModel = Model(inputs = model_inputs, outputs = conv6)
UNetModel.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy']);
print(UNetModel.summary())


# In[10]:

# Load the training data and labels
trainData,trainLabels = readSet(trainPaths)
validationData,validationLabels = readSet(validationPaths)


# In[ ]:

# Train model
UNetModel.fit(trainData, trainLabels,validation_data=(validationData,validationLabels), epochs=numberOfEpochs);


# In[112]:

# Save the model
UNetModel.save('networkModels/UNetModel.h5')


# In[23]:

# Use trained model to predict
predictAndSaveSet(testPaths, UNetModel);

