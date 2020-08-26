These are the part of the codes used for the article entitled "A Deep Learning
Approach for Assessment of Regional Wall Motion Abnormality from
Echocardiographic Images" for JACC CV imaging.  
  
license: MIT  
  
Before using the codes, please prepare image data at "./data_folder" dir.
  
./data_folder/train/Norm  
./data_folder/train/LAD  
./data_folder/train/LCX  
./data_folder/train/RCA  
  
./data_folder/test/Norm  
./data_folder/test/LAD  
./data_folder/test/LCX  
./data_folder/test/RCA  
  
Each dir should have echocardiographic images (.png is recommended and .jpg
acceptable) that contains endo-diastolic, mid-systolic, and endo-systolic phases. We put endo-diastolic image for Red color image channel, mid-systolic for Green and endo-systolic for Blue image channle with Python3.5 programming language with PIL and
numpy libraries.  
  
Please train with train_DCNN.py. By default it uses ResNet50. Training is performed using the image data in data_folder/train (80% for training, 20% for validation), and the post-trained internal parameter is saved in the current directory. Next, please run test_DCNN.py. This code reads the post-trained internal parameter saved in the current directory and it classifies the test data in data_folder/test (please move the post-trained internal parameter not used for analysis to another directory).  
  
These codes were used with  
   OS: Ubuntu 14.04LTS  
   Programming language: Python 3.5 Anaconda  
   Deep Learning library: tensorflow-gpu 1.4.1, Keras 2.1.5 
   CUDA toolkit 8.0, CuDNN v5.1 
   Python libraries: numpy 1.14.2, Pillow 5.0.0  
   
This code was written by a person who hold an e-mail account: taka4abe@gmail.com

ref: https://imaging.onlinejacc.org/content/13/2_Part_1/374.abstract
