import time
from sklearn.svm import LinearSVC, NuSVC
from loading import loadData

__author__ = 'Gabriel'

X_train, X_test,y_train, y_test = loadData("big")


print("Noyaux : lineaire")
start = time.time()

classifier = LinearSVC()
classifier.fit(X_train,y_train)


meanScore = classifier.score(X_test,y_test)

print("The mean score is : "+str(meanScore))
print("Elapsed time : "+str(time.time()-start))

#----------------------------------------------
print("Noyaux : rbf")
start = time.time()

classifier = NuSVC()
classifier.fit(X_train,y_train)


meanScore = classifier.score(X_test,y_test)

print("The mean score is : "+str(meanScore))
print("Elapsed time : "+str(time.time()-start))

