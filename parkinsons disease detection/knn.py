
from sklearn.decomposition import PCA
import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


df=pd.read_csv('parkinsons.csv')

features=df.loc[:,df.columns!='status'].values[:,1:]



labels=df.loc[:,'status'].values

# print(labels[labels==1].shape[0], labels[labels==0].shape[0])


scaler=MinMaxScaler((-1,1))

x=scaler.fit_transform(features)

y=labels
# x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=42)

# print(len(x_train[0]))

def Euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2            #(x1-x2)**2+(y1-y2)**2
    return distance**0.5

def get_class(x_train,y_train,test_data,k):
    pred_class = []
    for i in range(len(x_train)):
        dist = Euclidean_distance(x_train[i],test_data)
        temp = []
        temp.append(dist)
        temp.append(y_train[i])
        pred_class.append(temp)
    pred_class.sort()
    count_1 = 0
    count_0 = 0

    for i in range(k):
        if(pred_class[i][1]==1):
            count_1 += 1
        else:
            count_0 += 1

    if(count_1>count_0):
        return 1
    return 0

def classify(x_train,x_test,y_train,y_test,k):
    # x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=42)
    test_pred = []
    test_acctual = []
    for i in range(len(x_test)):
        c = get_class(x_train,y_train,x_train[i],k)
        test_pred.append(c)
        test_acctual.append(y_test[i])
    return test_pred
    count = 0
    for i in range(len(test_pred)):
        if(test_pred[i]==test_acctual[i]):
            count += 1
    return count/len(test_pred)

def KNN_Algo():
    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=42)
    k = [1,5,9,13]
    accuracy = []
    for i in k:
        pred = classify(x_train,x_test,y_train,y_test,i)
        count = 0
        for j in range(len(pred)):
            if(pred[j]==y_test[j]):
                count += 1
        acc = (count/len(pred))*100
        print("Accuracy for k = "+str(i)+" is "+str(acc) )
        print()
        accuracy.append(acc)
    print("maximum accuracy for knn is" ,max(accuracy))
    plt.bar(k,accuracy)
    plt.xticks(k)
    plt.xlabel("K values")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for KNN algorithm")
    plt.show()

KNN_Algo()
    
