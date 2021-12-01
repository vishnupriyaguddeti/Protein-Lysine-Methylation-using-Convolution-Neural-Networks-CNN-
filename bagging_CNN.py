
from sklearn.utils import resample
from sklearn.metrics import accuracy_score,confusion_matrix, precision_recall_curve
from matplotlib import pyplot
from numpy import mean
from numpy import std
import numpy
from numpy import array
from numpy import argmax
from NetworkCnn.simple_model import build_cnn
from Dataset.Loaddata import LoadData
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from  matplotlib import pyplot as plt
from keras import layers
import pandas as pd
import pickle
# evaluate a single mlp model
def label_prepare(label):
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(label)
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	return onehot_encoder.fit_transform(integer_encoded)
def evaluate_model(trainX, trainy, testX, testy):
	model = build_cnn()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model


	model.fit(trainX, trainy, epochs=20,batch_size=64, verbose=1)
	# evaluate the model
	_, test_acc = model.evaluate(testX, testy, verbose=1)
	return model, test_acc

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
	yhats = [model.predict(testX) for model in members]
	yhatsp =[model.predict_proba(testX) for model in members]
	yhats = array(yhats)
	yhatsp = array(yhatsp)
	# sum across ensemble members
	summed = numpy.sum(yhats, axis=0)
	summedp = numpy.sum(yhatsp, axis=0)
	# argmax across classes
	result = argmax(summed, axis=1)

	return result, summedp

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]
	# make prediction
	yhat, yhatprob = ensemble_predictions(subset, testX)
	print(confusion_matrix(testy.argmax(axis=1), yhat))
	predictions1 = yhatprob[:, 1]

	lr_precision, lr_recall, _ = precision_recall_curve(testy.argmax(axis=1), predictions1)
	pr50 = numpy.where(numpy.round(lr_precision, 1) == 0.5)
	sen = lr_recall[pr50]
	print(f'the score is : : {numpy.mean(sen)}')
	plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
	plt.show()
	# calculate accuracy
	return accuracy_score(testy.argmax(axis=1), yhat)


print('Loading data..............')
X = pd.read_csv ('train_X.csv').to_numpy()
X = numpy.expand_dims(X, axis=2)
y = pd.read_csv ('train_y.csv').to_numpy()
y = label_prepare(y)

test_X = pd.read_csv ('test_X.csv').to_numpy()
test_X = numpy.expand_dims(test_X, axis=2)
test_Y = pd.read_csv ('test_y.csv').to_numpy()
test_Y = label_prepare(test_Y)
print('Done Loading..................')

# multiple train-test splits
n_splits = 2
scores, members = list(), list()
for _ in range(n_splits):
	# select indexes
	ix = [i for i in range(len(X))]
	train_ix = resample(ix, replace=True, n_samples=15000)
	test_ix = [x for x in ix if x not in train_ix]
	# select data
	trainX, trainy = X[train_ix], y[train_ix]
	testX, testy = X[test_ix], y[test_ix]
	model, test_acc = evaluate_model(trainX, trainy, testX, testy)
	print('>%.3f' % test_acc)
	scores.append(test_acc)
	members.append(model)
# summarize expected performance
print('Estimated Accuracy %.3f (%.3f)' % (mean(scores), std(scores)))
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
#newX = numpy.expand_dims(newX, axis=2)
#newy = label_prepare(newy)
for i in range(1, n_splits+1):

	ensemble_score = evaluate_n_members(members, i, test_X, test_Y)
	#newy_enc = to_categorical(newy)
	_, single_score = members[i-1].evaluate(test_X, test_Y, verbose=0)
	print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
# plot score vs number of ensemble members
print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
x_axis = [i for i in range(1, n_splits+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()
filename = 'bagging_model.sav'
pickle.dump(members, open(filename, 'wb'))
