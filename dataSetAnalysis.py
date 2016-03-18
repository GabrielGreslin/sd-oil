from loading import loadData

__author__ = 'Gabriel'

X_train, X_test,y_train, y_test = loadData("big")

nbExempleBig = len(y_train)+len(y_test)
nbPositivExemple = sum(y_train)+sum(y_test)

print("Number of data point : " + str(nbExempleBig))
print("Number of oil data : " + str(nbPositivExemple) )
print("Number of not oil data : " + str(nbExempleBig-nbPositivExemple) )
print("Number of features : " + str(X_train.shape))