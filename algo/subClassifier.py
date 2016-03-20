import numpy
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import time
from sklearn.svm import SVC
from loading import loadDataPCA
from functools import reduce,partial

__author__ = 'Gabriel'


def voter(predicteursListe,Xtest):
    predictions = []
    for i in range(len(predicteursListe)):
        predictions.append(predicteursListe[i].predict(Xtest))

    predictionFinale = []

    for i in range(len(Xtest)):
        vote = sum([predictions[j][i] for j in range(len(predicteursListe))])

        if vote > 1.9:
            predictionFinale.append(1.0)
        elif vote < 0.1:
            predictionFinale.append(0.0)
        else:
            predictionFinale.append(0.5)

    return predictionFinale

def newSet(X_train,y_train,pred):

    numberHalfClaassified = sum(1 if x==0.5 else 0 for x in pred)

    print(X_train.shape)
    length,large = X_train.shape

    newXTrain = numpy.zeros((numberHalfClaassified,large))
    newYTrain = numpy.zeros((numberHalfClaassified,1))

    placeInNew=0
    for i,x in enumerate(X_train):
        if pred[i]== 0.5:
            newXTrain[placeInNew] = X_train[i]
            newYTrain[placeInNew] = y_train[i]
            placeInNew +=1
    print(newXTrain.shape)
    print(newYTrain.shape)
    return newXTrain,newYTrain


"""
The goal is to make random forest et SVM vote for the classification of oil or not, and see if it make the right decision
Then

"""

X_train, X_test,y_train, y_test = loadDataPCA("small")


start = time.time()

wc = SVC(probability =False,class_weight = {0.0:1,1.0:1.75}) #
wc.fit(X_train,y_train)
print("Elapsed time : "+str(time.time()-start))

rF = RandomForestClassifier(n_estimators=50,class_weight = {0.0:1,1.0:1.75})
rF.fit(X_train,y_train)
print("Elapsed time : "+str(time.time()-start))

pred = voter([wc,rF],X_train)

x_train_half, y_train_half = newSet(X_train,y_train,pred)

wc2 = RandomForestClassifier() #
wc2.fit(x_train_half,y_train_half)
print("Elapsed time : "+str(time.time()-start))

def newClassifier(x, c1,c2,c3):
    classification = c1(x) + c2(x)

    if classification > 1.9:
        return 1.0
    else:
        if c3(x) > 0.9:
            return 1.0
        else:
            return 0.0

c1 = lambda x:wc.predict(x)
c2 = lambda x:rF.predict(x)
c3 = lambda x:wc2.predict(x)
cls  = partial(newClassifier,c1=c1,c2=c2,c3=c3)

pred = [cls(x) for x in X_test]

if(y_test[0] == 1.0):
    t_names = ["Oil","Not Oil"]
else:
    t_names = ["Not Oil","Oil"]

cr = classification_report(y_test,pred,target_names=t_names)
print("Class report")
print(cr)

ac = confusion_matrix(y_test, pred, labels=[1.0,0.0])
print("Confusion matrix")
print(ac)
meanScore = accuracy_score(y_test,pred)
print("The mean score is : "+str(meanScore))
print("Elapsed time : "+str(time.time()-start))
