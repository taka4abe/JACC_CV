These are the part of the codes used for the article entitled "A Deep Learning
Approach for Assessment of Regional Wall Motion Abnormality from
Echocardiographic Images" for JACC CV imaging.

Before using this code, we should prepare image data at "/data_folder" dir.

/data_folder/train/Norm br
/data_folder/train/LAD
/data_folder/train/LCXD
/data_folder/train/RCA

/data_folder/test/Norm
/data_folder/test/LAD
/data_folder/test/LCX
/data_folder/test/RCA

Each dir should have echocardiographic images (.png is recommended and .jpg
acceptable) that contains endo-diastolic (ed), mid-systolic (mid), and endo-
systolic (ed) phases. We put ed for R (red) color image channel, mid for G,
and es for B image channle with Python3.5 programming language with PIL and
numpy libraries.

These codes were used with
   OS: Ubuntu 14.04LTS
   Programming language: Python 3.5 Anaconda
   Deep Learning library: tensorflow-gpu 1.4.1, Keras 2.1.5
   Python libraries: numpy 1.14.2, Pillow 5.0.0
