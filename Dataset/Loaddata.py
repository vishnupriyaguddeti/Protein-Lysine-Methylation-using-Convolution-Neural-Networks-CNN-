import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN, BorderlineSMOTE
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.utils import class_weight
from keras.regularizers import l2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class LoadData:
    def __init__(self, Datapath):
        self.Datapath = Datapath
        self.data = []

    def load(self):
        Data1 = pd.read_csv(self.Datapath[0], index_col=False).to_numpy()
        Labels1 = Data1[1:,-1]
        Data1=Data1[1:, 1:-1].astype(float)
        Data2 = pd.read_csv(self.Datapath[1], index_col=False).to_numpy()
        Labels2 = Data2[1:, -1]
        Data2 = Data2[1:, 1:-1].astype(float)

        Data=np.concatenate((Data1,Data2))
        Labels=np.concatenate((Labels1,Labels2))



        Labels = np.expand_dims(Labels, axis=1)
        self.data = np.expand_dims(Data, axis=2)
        return self.data, Labels
    def loadtest(self):
        Data1 = pd.read_csv(self.Datapath, index_col=False).to_numpy()
        Data1=Data1.astype(float)
        self.data = np.expand_dims(Data1, axis=2)
        return self.data
    def boxingblot(self,index):
        D = self.data
        plt.figure()
        plt.boxplot(D[:, index], 1)
        plt.show()
    def detect_outlier(self, index):
        outlier = []
        indices  = []
        mean = np.mean(self.data[:,index])
        standard = np.std(self.data[:,index])
        lowerlimit = mean - 3* standard
        upperlimit = mean + 3* standard
        for i, value in enumerate(self.data[:,index]):
            if(value > upperlimit or value < lowerlimit):
                outlier.append(value)
                indices.append(i)
        return outlier, indices, (lowerlimit,upperlimit)

    def normalization(self):
        M = np.mean(self.data,axis=0)
        S = np.std(self.data, axis=0)
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #dataset = np.squeeze(self.data)

        #scaler = scaler.fit(dataset)
        #print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        # normalize the dataset and print the first 5 rows
        #normalized = scaler.transform(dataset)
        #normalized = np.expand_dims(normalized, axis=2)
        return (self.data - M)/S
        #return normalized
    def check_missingdata(self):
        if(np.any(np.isnan(self.data))):
            print('[INFO] There is missing data............................ ')
        else:
            print('[INFO]  There is NO missing data ........................')

    def prepare_data(self, dataset, label, test_size = 0.25, test=0, noise=None, pca=None,no_component=10, valdition=None):
        print("===================Labels converted to binary ==============================")
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(label)
        print(integer_encoded)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        labels = onehot_encoder.fit_transform(integer_encoded)
        if(noise is not None):
            noises = np.random.normal(0,0.01,size=dataset.shape)
            dataset = np.concatenate((dataset,dataset + noises), axis=1)
        if(pca is not None):
            pca1 = PCA(n_components=no_component, random_state=42)
            dataset =np.squeeze(dataset)
            dataset = pca1.fit_transform(dataset)
            dataset = np.expand_dims(dataset,axis=2)
        if(not test and  valdition):
            X_train, X_test, y_train, y_test = train_test_split(dataset, integer_encoded, test_size=test_size, random_state=42)
            X_train = np.squeeze(X_train)

            X_test = np.squeeze(X_test)
            return X_train, X_test, y_train, y_test
        elif(not test and not valdition):
            X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=test_size, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
            #X_train = np.expand_dims(X_train, axis=2)
            #X_test = np.expand_dims(X_test, axis=2)
            return X_train, X_test, y_train, y_test, X_val, y_val
        else:
            X_test = np.expand_dims(dataset, axis=2)
            return X_test, labels


    def class_weight(self,y_train):
        N = sum(y_train[:,0])
        A = sum(y_train[:,1])
        T = N+A
        weight_for_0= (1/N)*(T)/2.0
        weight_for_1 = (1 / A) * (T) / 2.0
        class_weight = {0: weight_for_0, 1: weight_for_1}
        return class_weight

        #class_weights  = class_weight.compute_class_weight('balanced',
         #                                 np.unique(y_train),
          #                                y_train)
        #return class_weights

    def plotgraph(self,history):
        history_dict = history.history
        print(history_dict.keys())
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.clf()
        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']
        plt.plot(epochs, acc_values, 'bo', label='Training acc')
        plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_cm(self, y_test, predictions, p=0.5):
        cm = confusion_matrix(y_test.argmax(axis=1),
                                predictions.argmax(axis=1))
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f}'.format(p))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()

        print(' (True Negatives): ', cm[0][0])
        print(' (False Positives): ', cm[0][1])
        print(' (False Negatives): ', cm[1][0])
        print(' (True Positives): ', cm[1][1])
        print('specificity: ',cm[0][0]/(cm[0][0]+cm[0][1])*100)
        print('sensitivity: ',cm[1][1]/(cm[1][1]+cm[1][0])*100 )
        print('accuarcy : ', (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100)

    def scatterdata(self,X_train,y_train, index=1):
        y_train = np.where(y_train == 1)[1]
        counter = Counter(y_train)
        print(counter)
        # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = np.where(y_train == label)[0]
            plt.scatter(X_train[row_ix, 0], X_train[row_ix, 1], label=str(label))
        plt.legend()
        plt.show()
    def oversampling(self,X_train,y_train):
        oversample = SMOTE()
        #oversample = SVMSMOTE()
        #oversample = ADASYN()
        #oversample = BorderlineSMOTE()
        X_train=np.squeeze(X_train)
        if(y_train.shape[1]==2):
            y_train = np.where(y_train == 1)[1]
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        counter = Counter(y_train)
        print(counter)
        X_train = np.expand_dims(X_train, axis=2)
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        return X_train, y_train
    def detectandremoveoutlier_MCD(self, X_train, y_train, hist=None, plotbox=None, index=1):
        # Minimum Covariance Determinant and Extensions
        X_train = np.squeeze(X_train)
        if(hist is not None):
            self.plothist(X_train[:,index], title='Before remove outliers')

        ee = EllipticEnvelope(contamination=0.2,random_state=42)
        yhat = ee.fit_predict(X_train)
        # select all rows that are not outliers
        mask = yhat != -1
        X_train, y_train = np.expand_dims(X_train[mask, :], axis=2), y_train[mask]
        if (hist is not None):
            self.plothist(X_train[:, index], title='After remove outliers')
        if (plotbox is not None):
            plt.figure()
            plt.boxplot(X_train[:, index], 1)
            plt.show()
            outlier = []
            indices = []
            mean = np.mean(self.data[:, index])
            standard = np.std(self.data[:, index])
            lowerlimit = mean - 3 * standard
            uperlimit = mean + 3 * standard
            for i, value in enumerate(self.data[:, index]):
                if (value > uperlimit or value < lowerlimit):
                    outlier.append(value)
                    indices.append(i)

        return X_train, y_train

    def detectandremoveoutlier_IsolationForest(self, X_train, y_train, hist=None, index=1):
        # Minimum Covariance Determinant and Extensions
        X_train = np.squeeze(X_train)
        if (hist is not None):
            self.plothist(X_train[:, index], title='Before remove outliers')

        iso = IsolationForest(contamination=0.2,random_state=42)
        yhat = iso.fit_predict(X_train)
        # select all rows that are not outliers
        mask = yhat != -1
        X_train, y_train = np.expand_dims(X_train[mask, :], axis=2), y_train[mask]
        if (hist is not None):
            self.plothist(X_train[:, index], title='After remove outliers')


        return X_train, y_train

    def detectandremoveoutlier_IsolationForesttest(self, X_train):
        # Minimum Covariance Determinant and Extensions
        X_train = np.squeeze(X_train)
        iso = IsolationForest(contamination=0.2,random_state=42)
        yhat = iso.fit_predict(X_train)
        # select all rows that are not outliers
        mask = yhat != -1
        outlier_index = yhat == -1
        X_train = np.expand_dims(X_train[mask, :], axis=2)

        return X_train, outlier_index


    def plothist(self,d, title='Histogram'):
        # An "interface" to matplotlib.axes.Axes.hist() method
        n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(title)
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()

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











