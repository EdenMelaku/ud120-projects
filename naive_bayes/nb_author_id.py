#!/usr/bin/python
from tools.email_preprocess import preprocess
""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
t0=time()
clf.fit(features_train,labels_train )

print("training Time =", t0)

#########################################################


"""
import datetime

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import time

# Assigning features and label variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


#creating label encoder

lableEncoder= preprocessing.LabelEncoder()

#converting string lables into numbers

wheather_encoded=lableEncoder.fit_transform(weather)

print(wheather_encoded)


temp_encoded =lableEncoder.fit_transform(temp)

print(temp_encoded)

play_encoded=lableEncoder.fit_transform(play)

print(play_encoded)

features= list(zip(wheather_encoded, temp_encoded))

print (features)

model=GaussianNB()
now = datetime.datetime.now()
start_time=time.clock()
print("programm started running at ---- "+str(now.hour)+": "+str(now.minute)+": "+str(now.second) )

model.fit(features,play_encoded)

variable_init_Time = time.clock();
sec = (variable_init_Time - start_time) / 100
min = sec / 60
print("time elapsed for training= " + str(sec) + " seconds-------- or ------- " + str(min) + "   minutes")

#Predict Output
predicted= model.predict([[0,2]])
# 0:Overcast, 2:Mild
print ("Predicted Value:", predicted)

#importing datasets

from sklearn.datasets  import load_iris

data=load_iris()
print(data)

clf=GaussianNB()
clf.fit(data.data, data.target)
print("the lable names are  ",data.target_names)
predicted_val=clf.predict([[5.1,  3.5,  1.4,  0.2]])
print(predicted_val)
