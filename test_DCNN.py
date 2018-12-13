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

# finding posttrained data
for file_name in os.listdir('.'):
    root, ext = os.path.splitext(file_name)
    if ext == '.h5':
        posttrained_model = file_name
        NeuralNet, others = root.split('_', 1)
        break
    else:
        pass

print('')
if NeuralNet == 'VGG16':
    print('NeuralNetwork: VGG16.\n ILSVRC top-1 accuracy of 0.715\n')
    DCNN = keras.applications.vgg16.VGG16(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'VGG19':
    print('NeuralNetwork: VGG16.\n ILSVRC top-1 accuracy of 0.727\n')
    DCNN = keras.applications.vgg19.VGG19(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'ResNet50':
    print('NeuralNetwork: ResNet50.\n ILSVRC top-1 accuracy of 0.759\n')
    DCNN = keras.applications.resnet50.ResNet50(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'DenseNet201':
    print('NeuralNetwork: DenseNet201.\n ILSVRC top-1 accuracy of 0.770\n')
    DCNN = keras.applications.rdensenet.DenseNet201(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'InceptionV3':
    print('NeuralNetwork: InceptionV3.\n ILSVRC top-1 accuracy of 0.788\n')
    DCNN = keras.applications.inception_v3.InceptionV3(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'Xception':
    print('NeuralNetwork: Xception.\n ILSVRC top-1 accuracy of 0.790\n')
    DCNN = keras.applications.xception.Xception(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'IncResV2':
    print('NeuralNetwork: Inception-ResNet-V2.\n ILSVRC top-1 accuracy of 0.804\n')
    DCNN = keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=True, input_tensor=None, pooling=None, classes=1000)


model = Sequential()
model.add((DCNN))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation("softmax"))
opt = Adam(lr = 0.0001)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

model.load_weights(posttrained_model)
print("n\n_pre-trained weight loaded!!__")
print("    ", file_name, "\n")

# making data
image_list = []
label_list = []

for dir_name in os.listdir("data_folder/test"):
    dir_test = "data_folder/test/" + dir_name
    label = 0

    if dir_name == "LAD":
        label = 0
    elif dir_name == "LCX":
        label = 1
    elif dir_name == "RCA":
        label = 2
    elif dir_name == "Norm":
        label = 3

    for file_name in os.listdir(dir_test):
        label_list.append(label)
        filepath = dir_test + "/" + file_name
        image = np.array(Image.open(filepath).resize((224, 224)))
        result = model.predict_classes(np.array([image / 255]))
        print(filepath, "label:", label, "result:", result[0])
        if  label == 0:
            label_printer = 'LAD'
        elif label == 1:
            label_printer = 'LCX'
        elif label == 2:
            label_printer = 'RCA'
        elif label == 3:
            label_printer = 'Norm'
