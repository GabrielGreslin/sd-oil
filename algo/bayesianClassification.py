__author__ = 'Gabriel'


from loading import loadData

import time

from sklearn.naive_bayes import GaussianNB


X_train, X_test,y_train, y_test = loadData("big")

start = time.time()

gnb = GaussianNB()
gnb.fit(X_train, y_train)
#gnb.predict(iris.data)

meanScore = gnb.score(X_test,y_test)

print("The mean score is : "+str(meanScore))
print("Elapsed time : "+str(time.time()-start))