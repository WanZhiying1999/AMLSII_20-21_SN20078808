# AMLSII_20-21_SN20078808
This code is intended to solve two twitter sentiment analysis problem. Task A is to classify tweet of positive, negative or neutral. Task B is topic-based tweet sentiment analysis, which is two-point sacle:positive and negative.

### Environment
- Python==3.6.12
- Tensorflow==1.4
- h5py==2.10
- keras==2.2.0

### Directory Structure
Folder A and B contain the code for task A and B seperately.

- In the folder A, Preprocess_A.py
 is used to preprocess our original tweet for task A. trainA.py and testA.py contain the function to train and test our model for task A. lstm.py, cnn.py, bilstm.py are used as the comparation with our model. modelA.hdf5 save the trained model.
- In the folder B, Preprocess_B.py is used to preprocess our original tweet for task B. trainB.py and testB.py contain the function to train and test our model for task B.

Folder model contains the pre-trained GloVe model.

Folder dataset contains the training set and test set for task A and B. And it also save the clean data after preprocessing.


### Implementation Guide
Before you run the code, you can add images to the two data set folder. 
Download the pre-trained GloVe model 'glove.6B.100d.txt' in the release and put it in the folder Model.
Next, you can change the dataset path in main.py. 
Then, you can run the main.py to preprocess data, train and test the model.
If you want to compare other models with our method, you can run the cnn.py, lstm.py and bilstm.py directly in the folder A.

