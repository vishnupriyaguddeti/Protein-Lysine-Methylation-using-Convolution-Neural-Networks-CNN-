import numpy as np
import keras
from Dataset.Loaddata import LoadData
from collections import Counter
from imblearn.over_sampling import SMOTE, SVMSMOTE,ADASYN,BorderlineSMOTE
from sklearn.model_selection import train_test_split
from numpy import savetxt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def label_prepare(label):
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(label)

	return integer_encoded

def oversampling( X_train, y_train):
    oversample = SMOTE()
    #oversample = SVMSMOTE()
    # oversample = ADASYN()
    # oversample = BorderlineSMOTE()
    X_train = np.squeeze(X_train)
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    counter = Counter(y_train)
    print(counter)



    return X_train, y_train



datapath = 'C:/Users/nehar/OneDrive/Desktop/Carleton/Pattern Classif&Expermt Design (SEM) Winter 2021/project/csv_result-Descriptors_Training.csv'
datapathtest = 'C:/Users/nehar/OneDrive/Desktop/Carleton/Pattern Classif&Expermt Design (SEM) Winter 2021/project/csv_result-Descriptors_Calibration.csv'
datapathes=[datapath,datapathtest]
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import tensorflow as tf

    print(tf.__version__)
    print(keras.__version__)
    LD = LoadData(datapathes)
    Data, Label = LD.load()
    #Data1, Label1 =
    Normal=[]
    upnormal=[]
    for l in Label:
        if l == 'N':
            Normal.append(1)
        else:
            upnormal.append(1)
    print(f'[INFO] The frequency of normal is {sum(Normal)}')
    print(f'[INFO] The frequency of upnormal is {sum(upnormal)}')
    print(f'The data shape is {Data.shape}')
    print(f'The data shape is {Label.shape}')
    #LD.boxingblot(index=2)
    # detect outlier
    outlier, indeces, limit = LD.detect_outlier(index=2)
    print(outlier)
    print(indeces)

    Ndata = LD.normalization()
    LD.check_missingdata()
    X_train, X_test, y_train, y_test = train_test_split(Ndata, Label, test_size=0.3,
                                                        random_state=109)
    X_trainout, y_trainout = LD.detectandremoveoutlier_IsolationForest(X_train, y_train)
    X_valout, y_valout = LD.detectandremoveoutlier_IsolationForest(X_test, y_test)
    X_valout = np.squeeze(X_valout)
    y_trainout = label_prepare(y_trainout)
    y_valout = label_prepare(y_valout)
    
    # Apply pca
    Normal = []
    upnormal = []
    for l in y_valout:
        if l == '1':
            Normal.append(1)
        else:
            upnormal.append(1)
    print(f'[INFO] The frequency of normal is {sum(Normal)}')
    print(f'[INFO] The frequency of upnormal is {sum(upnormal)}')


    # oversample minority

    X_trainSample, y_trainSample = oversampling(X_trainout,y_trainout)
    savetxt('train_X.csv', X_trainSample, delimiter=',')
    savetxt('train_y.csv', y_trainSample, delimiter=',')
    savetxt('test_X.csv', X_valout, delimiter=',')
    savetxt('test_y.csv', y_valout, delimiter=',')
    print('done...................')