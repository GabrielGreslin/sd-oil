__author__ = 'Gabriel'

import sys
sys.path.append('../')

from loading import loadData, loadDataByCluster

import time

from sklearn.svm import SVC


for cluster in range(1, 5):
    X_train, X_test,y_train, y_test = loadDataByCluster(cluster, "big")

    start = time.time()

    classifier = SVC()
    classifier.fit(X_train,y_train)

    meanScore = classifier.score(X_test,y_test)

    print("Cluster : " + str(cluster))
    print("The mean score is : " + str(meanScore))
    print("")

print("Elapsed time : "+str(time.time()-start))
