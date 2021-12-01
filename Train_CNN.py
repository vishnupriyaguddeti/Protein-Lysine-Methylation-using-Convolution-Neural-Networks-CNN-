
import numpy as np
import keras
from keras.optimizers import Adam, SGD
from Dataset.Loaddata import LoadData
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping,LearningRateScheduler
from NetworkCnn.simple_model import build_cnn as Build
#from NetworkCnn.simple_model import build_cnn_api as Build
from keras import utils
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def poly_decay(epoch):
    NUM_EPOCHS = 100
    INIT_LR = 5e-3
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    # return the new learning rate
    return alpha
datapath = 'csv_result-Descriptors_Training.csv'
datapathtest = 'csv_result-Descriptors_Calibration.csv'
datapaths=[datapath, datapathtest]
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import tensorflow as tf

    print(tf.__version__)
    print(keras.__version__)
    LD = LoadData(datapaths)
    Data, Label = LD.load()
    #Data1, Label1 =
    Normal=[]
    Abnormal=[]
    for l in Label:
        if l == 'N':
            Normal.append(1)
        else:
            Abnormal.append(1)
    print(f'[INFO] The frequency of normal is {sum(Normal)}')
    print(f'[INFO] The frequency of upnormal is {sum(Abnormal)}')
    print(f'The data shape is {Data.shape}')
    print(f'The data shape is {Label.shape}')
    #LD.boxingblot(index=2)
    # detect outlier
    outlier, indices, limit = LD.detect_outlier(index=2)
    print(outlier)
    print(indices)

    Ndata = LD.normalization()
    LD.check_missingdata()
    X_train, X_test, y_train, y_test, X_val, y_val = LD.prepare_data(Ndata, Label)
    # detect and remove outlier
    X_trainout, y_trainout = LD.detectandremoveoutlier_IsolationForest(X_train, y_train)
    X_valout, y_valout = LD.detectandremoveoutlier_IsolationForest(X_val, y_val)

    # Apply pca


    class_weight = LD.class_weight(y_trainout)
    # oversample minority
    X_trainSample, y_trainSample = LD.oversampling(X_trainout,y_trainout)
    X_valSample, y_valSample = LD.oversampling(X_valout, y_valout)
    #LD.scatterdata(X_trainSample, y_trainSample)

    checkpoint = ModelCheckpoint("no1_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
    model = Build()
    #model = build_inception()
    model.summary()
    #opt = Adam(lr=1e-5,beta_1=0.9,beta_2=0.999)
    epochs = 10
    #opt = SGD(lr=1e-2, momentum=0.9, decay=1e-2 / epochs)
    opt = Adam(lr=1e-5)
    batch_size = 64
    #opt ='rmsprop'
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_trainout, y_trainout, epochs=epochs, batch_size=batch_size,
                      validation_data=(X_val,y_val),class_weight=class_weight,
                        callbacks=[checkpoint,early]) #LearningRateScheduler(poly_decay)
    LD.plotgraph(history)
    print("[INFO] evaluating network...")
    predictions = model.predict(X_test, batch_size=1,)
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1)))
    predictions1 = model.predict_proba(X_test, batch_size=1)
    predictions1 = predictions1[:, 1]
    LD.plot_cm(y_test, predictions)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test.argmax(axis=1), predictions1)
    pr50 = np.where(np.round(lr_precision, 1) == 0.5)
    sen = lr_recall[pr50]
    #scores.append(np.mean(sen))
    plt.plot(lr_recall, lr_precision)
    plt.show()
    model.save("FCnn_model")



