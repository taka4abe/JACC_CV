# -*- coding: utf-8 -*-

# This is the part of the codes used for the article entitled "A Deep Learning
# Approach for Assessment of Regional Wall Motion Abnormality from
# Echocardiographic Images" for JACC CV imaging.

# Before using this code, we should prepare image data at "/data_folder" dir.
#
# /data_folder/train/Norm
# /data_folder/train/LAD
# /data_folder/train/LCXD
# /data_folder/train/RCA
#
# /data_folder/test/Norm
# /data_folder/test/LAD
# /data_folder/test/LCX
# /data_folder/test/RCA
#
# Each dir should have echocardiographic images (.png is recommended and .jpg
# acceptable) that contains endo-diastolic (ed), mid-systolic (mid), and endo-
# systolic (es) phases. We put ed for R (red) color image channel, mid for G,
# and es for B image channle with Python3.5 programming language with PIL and
# numpy libraries.

# This code was used with
#    OS: Ubuntu 14.04LTS
#    Programming language: Python 3.5 Anaconda
#    Deep Learning library: tensorflow-gpu 1.4.1, Keras 2.1.5
#    Python libraries: numpy 1.14.2, Pillow 5.0.0
#
# If NeuralNet == "Xception":
#     this code takes about 4 min for training (100 epoches, 320 train/valid)
#     with core i7 6850K, RAM 256GB, NVMe SSD w 3.5" HDD, 1080ti.

import os, keras
import numpy as np
from datetime import datetime
from PIL import Image
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

# to select which neuralnetwork to use

#NeuralNet = 'VGG16'       # ILSVRC image classification top-1 accuracy of 0.715
#NeuralNet = 'VGG19'       # ILSVRC image classification top-1 accuracy of 0.727
NeuralNet = 'ResNet50'    # ILSVRC image classification top-1 accuracy of 0.759
#NeuralNet = 'DenseNet201' # ILSVRC image classification top-1 accuracy of 0.770
#NeuralNet = 'InceptionV3' # ILSVRC image classification top-1 accuracy of 0.788
#NeuralNet = 'Xception'    # ILSVRC image classification top-1 accuracy of 0.790
#NeuralNet = 'IncResV2'    # ILSVRC image classification top-1 accuracy of 0.804

# making training data
image_list = []
label_list = []

for dir_name in os.listdir("data_folder/train"):
    dir_train = "data_folder/train/" + dir_name
    label = 0

    if dir_name == "LAD":
        label = 0
    elif dir_name == "LCX":
        label = 1
    elif dir_name == "RCA":
        label = 2
    elif dir_name == "Norm":
        label = 3

    for file_name in os.listdir(dir_train):
        label_list.append(label)
        filepath = dir_train + "/" + file_name
        if NeuralNet == 'Xception':
            image = np.array(Image.open(filepath).resize((128, 128)))
        else:
            image = np.array(Image.open(filepath).resize((224, 224)))
        image_list.append(image / 255)

image_list = np.array(image_list)
label_list = to_categorical(label_list)

if __name__ == "__main__":
    #making neural network
    if NeuralNet == 'VGG16':
        print('NeuralNetwork: VGG16.\nILSVRC top-1 accuracy of 0.715')
        DCNN = keras.applications.vgg16.VGG16(
            include_top=True, input_tensor=None, pooling=None, classes=1000)
    elif NeuralNet == 'VGG19':
        print('NeuralNetwork: VGG16.\nILSVRC top-1 accuracy of 0.727')
        DCNN = keras.applications.vgg19.VGG19(
            include_top=True, input_tensor=None, pooling=None, classes=1000)
    elif NeuralNet == 'ResNet50':
        print('NeuralNetwork: ResNet50.\nILSVRC top-1 accuracy of 0.759')
        DCNN = keras.applications.resnet50.ResNet50(
            include_top=True, input_tensor=None, pooling=None, classes=1000)
    elif NeuralNet == 'DenseNet201':
        print('NeuralNetwork: DenseNet201.\nILSVRC top-1 accuracy of 0.770')
        DCNN = keras.applications.rdensenet.DenseNet201(
            include_top=True, input_tensor=None, pooling=None, classes=1000)
    elif NeuralNet == 'InceptionV3':
        print('NeuralNetwork: InceptionV3.\nILSVRC top-1 accuracy of 0.788')
        DCNN = keras.applications.inception_v3.InceptionV3(
            include_top=True, input_tensor=None, pooling=None, classes=1000)
    elif NeuralNet == 'Xception':
        print('NeuralNetwork: Xception.\nILSVRC top-1 accuracy of 0.790')
        DCNN = keras.applications.xception.Xception(
            include_top=True, input_tensor=None, pooling=None, classes=1000)
    elif NeuralNet == 'IncResV2':
        print('NeuralNetwork: Inception-ResNet-V2.\nILSVRC top-1 accuracy of 0.804')
        DCNN = keras.applications.inception_resnet_v2.InceptionResNetV2(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
    else:
        print('error, no neural network.')

    opt = Adam(lr = 0.0001)

    model = Sequential()
    model.add((DCNN))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    #training
    print('training')
    model.fit(image_list, label_list,
              epochs=1, batch_size=16, validation_split=0.2)

    #saving post-trained model
    prefix = datetime.now().strftime("%Y"+"_"+"%m%d"+"_"+"%H%M")
    save_name = NeuralNet + '_' + prefix + '.h5'
    model.save_weights(save_name)
    print('saving post-trained model:', save_name)
    print('finished training.')

print('finished: DCNN.py')
