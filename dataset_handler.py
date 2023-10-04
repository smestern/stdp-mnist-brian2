import numpy as np
import os.path
import pickle as pickle
import brian2 as b
from struct import unpack
from brian2 import *
from ni_interface.ni_brian2 import *
from sklearn.model_selection import train_test_split
import time
import copy as defaultcopy
import shutil

class mnistDataHandler(object):
    def __init__(self, N, labels=np.arange(10), train_test_split=0.5, MNIST_data_path="./MNIST/", self_retain=False):
        self.full_train_data = self.get_labeled_data(MNIST_data_path + 'training')
        self.test_full_test_data = self.get_labeled_data(MNIST_data_path + 'testing', bTrain=False)

        self.N = N
        self.labels = labels

    def load_data(self):

        #filter out the labels we don't want
        self.train_data = self.filter_data(self.full_train_data, self.labels )
        self.test_data = self.filter_data(self.test_full_test_data, self.labels )

        #filter down to the N samples, make sure the labels are evenly distributed
        self.train_data = self.filter_to_N_samples(self.train_data, self.N, self.labels )
        self.test_data = self.filter_to_N_samples(self.test_data, self.N, self.labels )

        return self.train_data, self.test_data

    def filter_to_N_samples(self, data, N, labels):
        #get the len of unique labels
        unique_labels = len(np.unique(data['y']))
        samples_per_label = int(N / unique_labels)
        print("samples_per_label: %i" % samples_per_label)
        filtered_data = {}
        filtered_data['x'] = np.zeros((N, data['rows'], data['cols']), dtype=np.uint8)
        filtered_data['y'] = np.zeros((N, 1), dtype=np.uint8)
        filtered_data['rows'] = data['rows']
        filtered_data['cols'] = data['cols']
        for i, label in enumerate(labels):
            label_idxs = np.where(data['y'] == label)[0]
            label_idxs = np.random.choice(label_idxs, samples_per_label, replace=False)
            filtered_data['x'][i*samples_per_label:(i+1)*samples_per_label] = data['x'][label_idxs]
            filtered_data['y'][i*samples_per_label:(i+1)*samples_per_label] = data['y'][label_idxs]
        #shuffle the data
        idxs = np.arange(N)
        np.random.shuffle(idxs)
        filtered_data['x'] = filtered_data['x'][idxs]
        filtered_data['y'] = filtered_data['y'][idxs]
        return filtered_data


    def filter_data(self, data, labels):
        #filter out the labels we don't want
        filtered_data = {}
        filtered_data['x'] = data['x'][np.isin(data['y'], labels).reshape(-1),:,:]
        filtered_data['y'] = data['y'][np.isin(data['y'], labels).reshape(-1)]
        filtered_data['rows'] = data['rows']
        filtered_data['cols'] = data['cols']
        return filtered_data


    @staticmethod
    def get_labeled_data(picklename, bTrain = True, MNIST_data_path="./MNIST/"):
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