__author__ = 'Gabriel'

from loading import loadData

import time

from sklearn.ensemble import RandomForestClassifier


X_train, X_test,y_train, y_test = loadData("big")

start = time.time()
#(n_estimators=30,max_depth=15)
randomF = RandomForestClassifier(n_estimators=100,max_depth=None)
randomF.fit(X_train,y_train)

meanScorerandomF = randomF.score(X_test,y_test)

print("The mean score is : "+str(meanScorerandomF))
print("Elapsed time : "+str(time.time()-start))