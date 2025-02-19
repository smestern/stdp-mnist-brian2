'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import pickle as pickle
from struct import unpack
from brian2 import *

CLASSES_SEEN = 2
#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]
        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * CLASSES_SEEN
    num_assignments = [0] * CLASSES_SEEN
    for i in range(CLASSES_SEEN):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    print( result_monitor.shape)
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in range(CLASSES_SEEN):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments

MNIST_data_path = os.getcwd()+'/MNIST/'
data_path = './activity_blanked/'
training_ending = '600'
testing_ending = '600'
SUM_TOTAL_TESTS = 600
start_time_training = 0
end_time_training = int(training_ending)
start_time_testing = 0
end_time_testing = int(testing_ending)

n_e = 40
n_input = 784
ending = ''

print( 'load MNIST')
training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)

print( 'load results')

training_result_monitor = np.load(data_path + 'resultPopVecs' + training_ending + ending + '.npy')
training_input_numbers = np.load(data_path + 'inputNumbers' + training_ending + '.npy')
testing_result_monitor = np.load(data_path + 'resultPopVecs' + testing_ending + '.npy')
testing_input_numbers = np.load(data_path + 'inputNumbers' + testing_ending + '.npy')
print( training_result_monitor.shape)

print( 'get assignments')
test_results = np.zeros((CLASSES_SEEN, end_time_testing-start_time_testing))
test_results_max = np.zeros((CLASSES_SEEN, end_time_testing-start_time_testing))
test_results_top = np.zeros((CLASSES_SEEN, end_time_testing-start_time_testing))
test_results_fixed = np.zeros((CLASSES_SEEN, end_time_testing-start_time_testing))
assignments = get_new_assignments(training_result_monitor[start_time_training:end_time_training],
                                  training_input_numbers[start_time_training:end_time_training])
print( assignments)
counter = 0
num_tests = end_time_testing / SUM_TOTAL_TESTS
sum_accurracy = [0] * int(num_tests)
while (counter < num_tests):
    end_time = min(end_time_testing, SUM_TOTAL_TESTS*(counter+1))
    start_time = SUM_TOTAL_TESTS*counter
    test_results = np.zeros((CLASSES_SEEN, end_time-start_time))
    print( 'calculate accuracy for sum')
    for i in range(end_time - start_time):
        test_results[:,i] = get_recognized_number_ranking(assignments,
                                                          testing_result_monitor[i+start_time,:])
    difference = test_results[0,:] - testing_input_numbers[start_time:end_time]
    correct = len(np.where(difference == 0)[0])
    incorrect = np.where(difference != 0)[0]
    sum_accurracy[counter] = correct/float(end_time-start_time) * 100
    print( 'Sum response - accuracy: ', sum_accurracy[counter], ' num incorrect: ', len(incorrect))
    counter += 1
print( 'Sum response - accuracy --> mean: ', np.mean(sum_accurracy),  '--> standard deviation: ', np.std(sum_accurracy))


#train a classifier on the training data
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score, f1_score

from sklearn.model_selection import train_test_split, permutation_test_score
from sklearn.model_selection import cross_val_score
x_train, x_test, y_train, y_test = train_test_split(testing_result_monitor[:,0].reshape(-1, 1), testing_input_numbers, test_size=0.3, stratify=testing_input_numbers, random_state=0)
clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(f"Cross Val Acc {np.nanmean(cross_val_score(clf, testing_result_monitor[:,0].reshape(-1, 1), testing_input_numbers, cv=5, scoring='balanced_accuracy'))}")

print( 'SVM accuracy: ', balanced_accuracy_score(y_test, y_pred))
print( 'SVM f1 score: ', f1_score(y_test, y_pred, average='macro'))
#permutation test
score, permutation_scores, pvalue = permutation_test_score(clf, x_test, y_test, scoring="f1_macro", cv=5, n_permutations=100, n_jobs=1)
print( "Classification score %s (pvalue : %s)" % (score, pvalue))

#save the cross val
scores = cross_val_score(clf, testing_result_monitor[:,0].reshape(-1, 1), testing_input_numbers, cv=5, scoring='balanced_accuracy')
np.savetxt(data_path+'cross_val_scores.csv', scores, delimiter=',')

#save the classifier
import pickle
filename = 'svm_classifier.sav'
pickle.dump(clf, open(filename, 'wb'))



#also run pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(testing_result_monitor)
x_pca = pca.transform(testing_result_monitor)
plt.scatter(x_pca[:,0], x_pca[:,1], c=testing_input_numbers)
plt.colorbar()



plt.figure()
#check the spiking preference of the first neuron
means = {}
for i in np.unique(testing_input_numbers):
    plt.hist(testing_result_monitor[testing_input_numbers==i,:], bins=100, alpha=0.5, label=str(i))
    means[i] = testing_result_monitor[testing_input_numbers==i,0]
plt.legend()
plt.figure()
plt.bar(means.keys(), [np.mean(x) for x in means.values()])
#save the dict
import pandas as pd
df = pd.DataFrame.from_dict(means, orient='index')
df.to_csv(data_path+'spiking_preference.csv')


plt.show()