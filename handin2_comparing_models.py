'''
Created on 1 Dec 2017

@author: pings
'''
import pickle, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from bayes_classifier import NBC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 

''' Comparing NBC to Logistic Regression '''
def average_performance(Classifier, Xtrain, ytrain, Xtest, ytest, k=10, classifier_args={}, trials=1000):
    '''
    Average the performance of ``Classifier`` over varying percentages of the training data over a number of ``trials``.
    
    Returns an np array of size k of the average test accuracies obtained.
    '''
    trial_runs = np.zeros((trials, k))
    for n in range(trials):
        # populating the matrix with values obtained from the nth run
        trial_runs[n] = _evaluate_k_classifiers(Classifier, Xtrain, ytrain, Xtest, ytest, k, classifier_args)
        Xtrain, ytrain = _shuffle_arrays(Xtrain, ytrain)
        
    return np.mean(trial_runs, 0)

def _shuffle_arrays(a, b):
    '''
    Shuffles 2 arrays in unison
    Returns the shuffled version of the arrays
    '''
    assert(len(a) == len(b))
    p = np.random.permutation(len(a))
    
    return a[p], b[p]

def _evaluate_k_classifiers(Classifier, X, y, Xtest, ytest, k=10, classifier_args={}):
    '''
    Takes in a classifier Class (interfaced similarly to sklearn), as well as some training and test data.
    Train and evaluate `k` copies of the classifier, with the ith using i*100/k% of the training data. 
    
    Returns an np array of length k, with the test accuracy of each of the `k` classifiers.
    '''
    accuracies = []
    for i in range(1,k+1):
        # assign a proportion of the data to the classifier
        Xtrain, _, ytrain, _ = train_test_split(X, y, train_size=i/k-1e-4)
        accuracies.append(_evaluate_classifier(Classifier(**classifier_args), Xtrain, ytrain, Xtest, ytest))
        
    return np.array(accuracies)

def _evaluate_classifier(classifier, X, y, Xtest, ytest):
    '''
    Takes in a classifier (with the interface of sklearn classifiers), and some data + test data in form of numpy arrays
    Returns the test accuracy of the model on the test data.
    '''
    classifier.fit(X, y)
    yhat = classifier.predict(Xtest)
    return np.mean(yhat == ytest)

def plot_2line_graph(y1, y2):
    '''
    Plot a 2-line graph with data y1, y2. (Assumes y1, y2 have the same size)
    '''
    x = range(y1.size)
    plt.plot(x, y1, label='nbc'); plt.plot(x, y2, label='lr')
    plt.legend()
    plt.xlabel('proportion of data used'); plt.ylabel('test accuracy')

''' Iris Dataset '''
# obtaining the Iris dataset and doing 80/20 train/test split
iris = load_iris()
X, y = iris['data'], iris['target']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

# getting the accuracies for both classifiers
nbc_iris_arguments = {'feature_types': ['c']*4, 'num_classes': 3}
nbc_iris_accuracies = average_performance(NBC, Xtrain, ytrain, Xtest, ytest, classifier_args=nbc_iris_arguments)
lr_iris_accuracies = average_performance(LogisticRegression, Xtrain, ytrain, Xtest, ytest)

# plot the graph (subplot(nrow, ncol, plotnum))
plt.subplot(1, 2, 1)
plt.title('iris')
plot_2line_graph(nbc_iris_accuracies, lr_iris_accuracies)

''' 2016 Voting Dataset ''' 
# loading & splitting the dataset
X, y = pickle.load(open('./data/voting.pickle', 'rb'))
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

# getting the accuracies for both classifiers
nbc_voting_arguments = {'feature_types': ['b']*16, 'num_classes': 2}
nbc_voting_accuracies = average_performance(NBC, Xtrain, ytrain, Xtest, ytest, classifier_args=nbc_voting_arguments)
lr_voting_accuracies = average_performance(LogisticRegression, Xtrain, ytrain, Xtest, ytest)

# plot the graph
plt.subplot(1, 2, 2)
plt.title('voting')
plot_2line_graph(nbc_voting_accuracies, lr_voting_accuracies)

plt.show()
