import os
import numpy
from sklearn import decomposition
from sklearn.cross_validation import train_test_split


__author__ = 'Gabriel'

def loadData(dataType='small') :

    dirPath = ''
    path = dirPath + 'data/Small_F_oil_data_SUPAERO.csv'
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath

    if dataType == 'small':
        fileNameData = dirPath + 'data/Small_F_oil_data_SUPAERO.csv'
        fileNameLabel = dirPath + 'data/Small_F_oil_label_SUPAERO.csv'
        x = numpy.genfromtxt(fileNameData, delimiter=",")
        y = numpy.genfromtxt(fileNameLabel, delimiter=",")
    elif dataType == 'big':
        fileNameData = dirPath + 'data/F_data_oil_SUPAERO.csv'
        fileNameLabel = dirPath + 'data/F_label_oil_SUPAERO.csv'
        x = numpy.genfromtxt(fileNameData, delimiter=",")
        y = numpy.genfromtxt(fileNameLabel, delimiter=",")

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def loadDataVal(dataType='small'):

    dirPath = ''
    path = dirPath + 'data/Small_F_oil_data_SUPAERO.csv'
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath

    if dataType == 'small':
        fileNameData = dirPath + 'data/Small_F_oil_data_SUPAERO.csv'
        fileNameLabel = dirPath + 'data/Small_F_oil_label_SUPAERO.csv'
        x = numpy.genfromtxt(fileNameData, delimiter=",")
        y = numpy.genfromtxt(fileNameLabel, delimiter=",")
    elif dataType == 'big':
        fileNameData = dirPath + 'data/F_data_oil_SUPAERO.csv'
        fileNameLabel = dirPath + 'data/F_label_oil_SUPAERO.csv'
        x = numpy.genfromtxt(fileNameData, delimiter=",")
        y = numpy.genfromtxt(fileNameLabel, delimiter=",")

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42)
    return X_train, X_test, X_val, y_train, y_val, y_test

def loadDataPCA(dataType='small'):

    dirPath = ''
    path = dirPath + 'data/Small_F_oil_data_SUPAERO.csv'
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath
    if not os.path.isfile(dirPath + path) :
        dirPath = '../' + dirPath

    if dataType == 'small':
        fileNameData = dirPath + 'data/Small_F_oil_data_SUPAERO.csv'
        fileNameLabel = dirPath + 'data/Small_F_oil_label_SUPAERO.csv'
        x = numpy.genfromtxt(fileNameData, delimiter=",")
        y = numpy.genfromtxt(fileNameLabel, delimiter=",")
    elif dataType == 'big':
        fileNameData = dirPath + 'data/F_data_oil_SUPAERO.csv'
        fileNameLabel = dirPath + 'data/F_label_oil_SUPAERO.csv'
        x = numpy.genfromtxt(fileNameData, delimiter=",")
        y = numpy.genfromtxt(fileNameLabel, delimiter=",")

    pca = decomposition.PCA(n_components=44, copy=True, whiten=False)
    pca.fit(x)
    print(pca.explained_variance_ratio_)

    x_transformed = pca.transform(x)
    print(str(x_transformed.shape))

    X_train, X_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

"""
Force the proportion of true value in a training set
"""
def forceProportion(values,labels,proportionClass1 = 0.5):
    assert proportionClass1 != 0.0

    total = len(labels)
    numberClass1 = sum(labels)
    numberClass0 = total-numberClass1

    newTot = int(numberClass1/proportionClass1)
    class0NumberToKeep = int((1-proportionClass1) * newTot)
    assert class0NumberToKeep + numberClass1 == newTot

    width_array = len(values[0])

    newValues = numpy.zeros((newTot,width_array))
    newLabels = numpy.zeros((newTot))

    i = numberClass1
    j = class0NumberToKeep

    currentLineRead = 0
    currentLineWrite = 0

    while max(i,j) > 0:
        if labels[currentLineRead] == 1.0:
            if(i>0):
                newValues[currentLineWrite] = values[currentLineRead]
                newLabels[currentLineWrite] = labels[currentLineRead]
                i -=1
                currentLineWrite +=1
        else:
            if(j>0):
                newValues[currentLineWrite] = values[currentLineRead]
                newLabels[currentLineWrite] = labels[currentLineRead]
                j -=1
                currentLineWrite +=1

        currentLineRead +=1

    return newValues,newLabels
