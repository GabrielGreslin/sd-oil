from sklearn.metrics import classification_report, confusion_matrix

__author__ = 'Gabriel'


from loading import loadData, loadDataPCA

import time

from sklearn.svm import SVC



X_train, X_test,y_train, y_test = loadDataPCA("big")

start = time.time()

classifier = SVC(probability =False,class_weight = {0.0:1,1.0:1.75}) #
classifier.fit(X_train,y_train)

meanScore = classifier.score(X_test,y_test)

pred = classifier.predict(X_test)

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


print("The mean score is : "+str(meanScore))
print("Elapsed time : "+str(time.time()-start))

