from A import Preprocess_A,trainA,testA
from B import Preprocess_B,trainB,testB
#===============================================================================================================================
#data preprocess:preprocess the original tweet and save the clean data as csv file for further use
Preprocess_A.Tweet_A('Datasets\\Train\\SemEval2017-task4-dev.subtask-A.english.INPUT.txt','Datasets\\Train\\train_A.csv')
Preprocess_A.Tweet_A('Datasets\\Test\\twitter-2015test-A.txt','Datasets\\Test\\test_A.csv')
Preprocess_B.Tweet_B('Datasets\\train\\SemEval2017-task4-dev.subtask-BD.english.INPUT.txt','Datasets\\Train\\train_B.csv')
Preprocess_B.Tweet_B('Datasets\\Test\\twitter-2015testBD.txt','Datasets\\Test\\test_B.csv')

#===============================================================================================================================
#task A
#train our model
acc_A_train, val_acc_A_train = trainA.train_A('Datasets\\train\\train_A.csv')

#test our model on seperate test set
acc_A_test,f1_A = testA.test_A('Datasets\\Test\\test_A.csv')

#===============================================================================================================================
#task B
acc_B_train, val_acc_B_train = trainB.train_B('Datasets\\train\\train_B.csv')

acc_B_test, f1_B = testB.test_B('Datasets\\Test\\test_B.csv')

# ======================================================================================================================
## Print out your results with following format:
print('TaskA:Train accuray:{},validation accuracy:{},test accuracy:{},f1 score of model:{};TaskB:Train accuray:{},validation accuracy:{},test accuracy:{},f1 score of model:{};'.format(acc_A_train, val_acc_A_train, acc_A_test,f1_A,
                                                        acc_B_train, val_acc_B_train, acc_B_test, f1_B))

