#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
import datetime
import sys
sys.path.append("../tools/")
from tools.email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


#########################################################
clf=DecisionTreeClassifier()
now = datetime.datetime.now()
import time
start_time=time.clock()
print("programm started running at ---- "+str(now.hour)+": "+str(now.minute)+": "+str(now.second) )

clf.fit(features_train,labels_train)

variable_init_Time = time.clock()
sec = (variable_init_Time - start_time) / 100
min = sec / 60
print("time elapsed for training= " + str(sec) + " seconds-------- or ------- " + str(min) + "   minutes")

lables=clf.predict(features_test)
print("accuracy = ",metrics.accuracy_score(labels_test,lables))
#plot=tree.plot_tree(clf.fit(features_train,labels_train))


