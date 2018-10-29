This repository contains the implementation of a Face detection model on AFLW dataset.

The whole model is implemented in PyTorch
It can be installed using the following command if conda is installed in the
system: conda install pytorch torchvision -c pytorch
If conda is not installed, other installation commands can be found from http://pytorch.org/


Dataset

The model is trained on AFLW dataset with 10000 face images and 10000 non face image.
For testing, 100 face and non face images are taken.
SQLite is used to extract images from the AFLW database.


Dataset Loader

The code data.py is used to generate the data from the database. 
The code making_data.py is used to generate csv file which has all the image names on it’s first column and labels on it’s second column where 1
represents a face image and 0 represents a non face image.
Then to make this data compatible with PyTorch and to load it, the code Face.py and Face_test.py can be sued which provides an interface for the data to be loaded using the default torch.utils.data.DataLoader in PyTorch.
