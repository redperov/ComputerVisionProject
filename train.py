import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical, Sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, MaxPool2D, AveragePooling2D, Dropout
from keras.losses import SparseCategoricalCrossentropy
#from kerastuner.tuners import RandomSearch
from keras.models import load_model
import h5py
import shelve
import pdb

# The paths to the h5 files
raw_train_data_path = "/content/SynthText.h5"
raw_validation_data_path = "/content/SynthText_val.h5"

# The path to which the modified data will be stored
train_data_path = "train"
validation_data_path = "validation"
train_and_validation_data_path = "train_and_validation"

# Parameters
SIZE = 400


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, shelve_db, list_IDs, batch_size=32, dim=(SIZE, SIZE), n_channels=1,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.shelve_db = shelve_db
        self.dim = dim
        self.shelve_db = shelve_db
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            item = self.shelve_db[ID]
            X[i,] = item["char_image"]
            y[i] = item["label"]

            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            # y[i] = self.labels[ID]

        return X, to_categorical(y, num_classes=self.n_classes)


def strings_to_ints(labels):
    ### Universal list of colors
    total_labels = ["Ubuntu Mono", "Skylark", "Sweet Puppy"]

    ### map each color to an integer
    mapping = {}
    for x in range(len(total_labels)):
        mapping[total_labels[x]] = x

    # integer representation
    for x in range(len(labels)):
        labels[x] = mapping[labels[x]]

    return labels


def count_instances(path):
    instances = {"Ubuntu Mono": [], "Skylark": [], "Sweet Puppy": []}

    with shelve.open(path) as shelve_db:
        for key in shelve_db:
            class_name = key.split("_")[0]
            array = instances[class_name]
            array.append(key)

    for instance in instances:
        print(f"{instance}: {len(instances[instance])}")
    return instances


def get_filenames_and_labels(path):
    filenames = []
    fonts = []

    with shelve.open(path) as shelve_db:
        for key in shelve_db:
            font = key.split("_")[0]
            filenames.append(key)
            fonts.append(font)

    filenames = np.array(filenames)
    labels = np.array(strings_to_ints(fonts))

    return filenames, labels


if __name__ == '__main__':
    # instances = count_instances(train_and_validation_data_path)
    filenames, labels = get_filenames_and_labels(train_and_validation_data_path)

    # Split the data into train and test sets
    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(filenames, labels,
                                                                          train_size=0.8,
                                                                          random_state=42,
                                                                          stratify=labels)

    model = Sequential([
        Flatten(),
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(3, activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

    params = {'dim': (400, 400),
              'batch_size': 64,
              'n_classes': 3,
              'n_channels': 1,
              'shuffle': True}

    with shelve.open(train_and_validation_data_path) as shelve_db:
        training_generator = DataGenerator(shelve_db, X_train_filenames, **params)
        validation_generator = DataGenerator(shelve_db, X_val_filenames, **params)
        model.fit(training_generator,
                  epochs=10,
                  validation_data=validation_generator,
                  use_multiprocessing=True,
                  workers=6)