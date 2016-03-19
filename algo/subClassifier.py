from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import time
from sklearn.svm import SVC
from loading import loadDataPCA

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

"""
The goal is to make random forest et SVM vote for the classification of oil or not, and see if it make the right decision
Then

"""

X_train, X_test,y_train, y_test = loadDataPCA("big")


start = time.time()

wc = SVC(probability =False,class_weight = {0.0:1,1.0:1.75}) #
wc.fit(X_train,y_train)
print("Elapsed time : "+str(time.time()-start))

rF = RandomForestClassifier(n_estimators=50,class_weight = {0.0:1,1.0:1.75})
rF.fit(X_train,y_train)
print("Elapsed time : "+str(time.time()-start))


pred = voter([wc,adada,rF],X_test)

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
