__author__ = 'Gabriel'

from loading import loadData

import time

from sklearn.ensemble import AdaBoostClassifier


X_train, X_test,y_train, y_test = loadData("big")

start = time.time()

classifier = AdaBoostClassifier()
classifier.fit(X_train,y_train)

meanScore = classifier.score(X_test,y_test)

print("The mean score is : "+str(meanScore))
print("Elapsed time : "+str(time.time()-start))