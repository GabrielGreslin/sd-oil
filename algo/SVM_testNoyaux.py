__author__ = 'Gabriel'

import sys
sys.path.append('../')

from loading import loadData

import time

from sklearn.svm import SVC


X_train, X_test,y_train, y_test = loadData("big")

#Test of the following kernels linear, poly, rbf, sigmoid
#----------------------------------------------
""" Not working, too slow
print("Noyaux : linear")
start = time.time()

classifier = SVC(kernel="linear")
classifier.fit(X_train,y_train)


meanScore = classifier.score(X_test,y_test)

print("The mean score is : "+str(meanScore))
print("Elapsed time : "+str(time.time()-start))
"""
#----------------------------------------------
print("Noyaux : poly")
start = time.time()

classifier = SVC(kernel="poly")
classifier.fit(X_train,y_train)


meanScore = classifier.score(X_test,y_test)

print("The mean score is : "+str(meanScore))
print("Elapsed time : "+str(time.time()-start))

#----------------------------------------------
print("Noyaux : rbf")
start = time.time()

classifier = SVC(kernel="rbf")
classifier.fit(X_train,y_train)


meanScore = classifier.score(X_test,y_test)

print("The mean score is : "+str(meanScore))
print("Elapsed time : "+str(time.time()-start))

#----------------------------------------------
print("Noyaux : sigmoid")
start = time.time()

classifier = SVC(kernel="sigmoid")
classifier.fit(X_train,y_train)


meanScore = classifier.score(X_test,y_test)

print("The mean score is : "+str(meanScore))
print("Elapsed time : "+str(time.time()-start))
