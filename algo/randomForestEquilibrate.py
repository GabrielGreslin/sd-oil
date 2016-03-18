__author__ = 'Gabriel'

from loading import loadData, forceProportion

import time

from sklearn.ensemble import RandomForestClassifier


X_train, X_test,y_train, y_test = loadData("big")

#Equilibration du set de training

X_train,y_train = forceProportion(X_train,y_train,0.5)
nbExempleBig = len(y_train)+len(y_test)
nbPositivExemple = sum(y_train)+sum(y_test)

print("Number of data point : " + str(nbExempleBig))
print("Number of oil data : " + str(nbPositivExemple) )
print("Number of not oil data : " + str(nbExempleBig-nbPositivExemple) )

#--------------------------------


start = time.time()

randomF = RandomForestClassifier(n_estimators=20,max_depth=10)
randomF.fit(X_train,y_train)

meanScorerandomF = randomF.score(X_test,y_test)

print("The mean score is : "+str(meanScorerandomF))
print("Elapsed time : "+str(time.time()-start))