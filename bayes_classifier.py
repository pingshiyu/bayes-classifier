'''
Created on 30 Nov 2017

@author: pings
'''

'''
Implementation of Naive-bayes classifier in Python from scratch.

This will have an interface as follows:
nbc = NBC(feature_types=['b', 'c', 'b'], num_classes=4)
where 'b' denotes a binary-data type and a 'c' denotes a continuous datatype.

We assume that the classes are numbers from the set [0, 1, ..., num_classes-1]
'''

import numpy as np

class NBC():
    def __init__(self, feature_types, num_classes, domains=None):
        '''
        ``feature_types``: list of chars, specify what sort of types of features we will be getting;
        ``num_classes``: integer, specifies the number of output classes;
        ``domains``: list of integer, specifies the domain of each feature. None for continuous features.
                     i.e. domains[2] == 5 => 5th feature takes values from 0 to 5-1
        '''
        self._feature_types = feature_types
        self._num_classes = num_classes
        if not domains:
            # no domains specified - use binary by default for discrete features
            self._domains = [2 if t != 'c' else None for t in feature_types]
        else:
            self._domains = domains
        # parameters for our model, (a list of parameters for each feature)
        self._params = None

    def fit(self, X, y):
        '''
        Fit the model according to the data given, `X` and `y`
        '''
        # slice ``X`` into its features, i.e. X_slice[i] gives X's ith feature
        X_slice = X.T

        # obtain the parameters for each feature according to their type (continuous or discrete)
        self._params = [self._gaussian_feature_statistic(x, y) if self._feature_types[i] == 'c'
                        else self._bernoulli_feature_statistic(x, self._domains[i], y) for i, x in enumerate(X_slice)]

    def predict(self, X):
        '''
        Return the prediction on ``X`` based on the current model

        Decision is based on taking the maximal log-probability class of ``X`` based on our current model
        '''
        # again X_slice[i] gives X's ith feature
        X_slice = X.T

        # here total_log_prob[i, c] = log(P(x[i] | y[i] = c, params)
        total_log_prob = np.zeros((len(X), self._num_classes))
        for i, x in enumerate(X_slice):
            # loop over the features of X
            if self._feature_types[i] == 'c':
                # continuous feature
                gaussian_params = self._params[i]
                mean, var = gaussian_params.T
                log_prob = -0.5*np.log(2*np.pi*var) - (x[np.newaxis].T - mean)**2 / (2*(var**2))
            else:
                # discrete feature
                # here mu[c, l] = P(x = l | y = c)
                mu = self._params[i]
                # picks out the columns corresponding to the probability distribution of x over C
                log_prob = np.log(mu[:, x.astype(int)]).T

            total_log_prob += log_prob

        # return prediction of highest probability class for each datapoint in X
        return np.argmax(total_log_prob, 1)

    def _gaussian_feature_statistic(self, x, classes):
        '''
        x: 1d numpy array, the values of the feature x
        classes: 1d numpy array, the values of the output classes

        Calculate the Gaussian statistic for the continuous feature ``x``
        Outputs an array of the mean and variance for each class, i.e. mu_c and sigma_c for each
        c in ``classes``
        '''
        # params for the continuous feature, each class has associated [class_mean, class_variance]
        params = np.zeros((self._num_classes, 2))
        for c in range(self._num_classes):
            # isolate the features belonging to the c class - 'given class == c'
            x_c = x[classes == c]
            params[c, 0] = np.mean(x_c)
            params[c, 1] = max(1e-5, np.var(x_c)) # ensure it's not 0

        return params

    def _bernoulli_feature_statistic(self, x, domain, classes, l=1):
        '''
        x: 1d numpy array, the jth feature of a dataset
        domain: integer, the range of values ``x`` datapoints can take
        classes: 1d numpy array, the output classes corresponding to each datapoint
        l: integer, laplace smoothing constant

        Calculate the parameters for the categorical feature. This is based on a Bernoulli distribution.
        To calculate this we simply calculate p(x = l | y = c) empirically to determine the parameters.
        i.e. #(x == l and y == c) / #(y == c)
        Returns an array of (C, K) shape, C being the number of classes, K being the size of the domain for feature x.
        '''
        # classes_count[c] is the number of elements in ``classes`` equalling to c
        classes_count = np.array([np.sum(classes == c) for c in range(self._num_classes)]) + 2*l

        # conditional_count[c, l] == #(y == c && x == l)
        conditional_count = np.array(
            [np.array([np.sum((classes == c) * (x == l)) for l in range(domain)]) + l
            for c in range(self._num_classes)])

        # conditional_probs[c, l] = P(x = l | y = c) = #(y == c && x == l) / #(y == c)
        conditional_probs = conditional_count / classes_count[np.newaxis].T
        return conditional_probs

if __name__ == '__main__':
    '''
    testing:
    '''
    '''with open('./data/voting-full.pickle', 'rb') as f:
        data = pickle.load(f)
        print(data[1])'''

    ''''x = np.array([1, 2, 1, 5, 6, 7, 8])
    var = np.array([1, 1, 0.5, 1, 2])
    mean = np.array([0, 4, 7, 12, 4])
    log_prob = -0.5*np.log(2*np.pi*var) - ((x[np.newaxis].T - mean)**2) / (2*(var**2))

    select = np.array([1, 3, 0, 4, 1, 0, 0, 0])
    print(log_prob[:, select])'''
