'''
Created on 30 Nov 2017

@author: pings
'''
from bayes_classifier import NBC
import pickle, numpy as np

''' unpickling the data '''
X, y = pickle.load(open('./data/voting.pickle', 'rb'))

''' splitting into training and testing sets '''
N, D = X.shape
N_train = int(0.8 * N)
N_test = N - N_train
X_train, y_train = X[:N_train], y[:N_train]
X_test, y_test = X[N_train:], y[N_train:]

nbc = NBC(feature_types=['b']*16, num_classes=2)
nbc.fit(X_train, y_train)
yhat = nbc.predict(X_test)
test_accuracy = np.mean(yhat == y_test)

# with voting.pickle the test accuracy was 97.87%
print('NBC accuracy on voting.pickle:', test_accuracy)

'''
Handin 1:

For Logistic regression of the sklearn package: 
If we want to use lambda as our parameter, then we should set C to be 1/(2*lambda) according to the implementation in scipy.
'''