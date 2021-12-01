import numpy as np
from keras import models
from Dataset.Loaddata import LoadData
from collections import Counter
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN, BorderlineSMOTE
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
tf.compat.v1.disable_eager_execution()

def label_prepare(label):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(label)

    return integer_encoded


def oversampling(X_train, y_train):
    oversample = SMOTE()
    X_train = np.squeeze(X_train)
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    counter = Counter(y_train)
    print(counter)

    return X_train, y_train


datapathes = 'C:/Users/nehar/OneDrive/Desktop/Carleton/Pattern Classif&Expermt Design (SEM) Winter 2021/Assigment/pythonProject/Blind_Test_features.csv'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    LD = LoadData(datapathes)
    Data = LD.loadtest()
    print(f'The data shape is {Data.shape}')



    test_X = LD.normalization()
    LD.check_missingdata()
    model1 = models.load_model('Cnn_model')
    print("[INFO] evaluating network without remove outlier...")
    #test_XX = np.expand_dims(test_X, axis=2)
    predictions = model1.predict(test_X, batch_size=1)
    predictions = predictions.argmax(axis=1)
    predictions_prob = model1.predict_proba(test_X, batch_size=1)
    Normal = []
    upnormal = []
    for l in predictions:
        if l == 1:
            Normal.append(1)
        else:
            upnormal.append(1)
    print(f'[INFO] The frequency of normal is {sum(Normal)}')
    print(f'[INFO] The frequency of upnormal is {sum(upnormal)}')
    # oversample minority
    np.savetxt('Result_without_removeoutlier', predictions, delimiter=',',fmt='%d')
    np.savetxt('Result_without_removeoutlier_prob', predictions_prob, delimiter=',')

    # Remove the outlier
    X_trainout, outlier_index = LD.detectandremoveoutlier_IsolationForesttest(test_X)
    outlist = list()
    outlistprob = list()
    for k, logicvalue  in enumerate(outlier_index):
        if (logicvalue):
            outlist.append('Outlier')
            outlistprob.append('Outlier')
        else:
            test_one=test_X[k,:,:];
            test_one = np.expand_dims(test_one,axis=0)
            predict2 = model1.predict(test_one, batch_size=1)
            predict = predict2.argmax(axis=1)
            outlist.append(predict)
            outlistprob.append(predict2)

    # open file
    with open('Result_with_removeoutlier.txt', 'w+') as f:

        # write elements of list
        for items in outlist:
            f.write('%s\n' % items)
        print("File written successfully")
    # close the file
    f.close()

    with open('Result_with_removeoutlierprob.txt', 'w+') as f:

        # write elements of list
        for items in outlistprob:
            f.write('%s\n' % items)
        print("File written successfully")
    # close the file
    f.close()

    print('done...................')