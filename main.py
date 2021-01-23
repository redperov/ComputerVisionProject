import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical, Sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, MaxPool2D, AveragePooling2D, Dropout
from keras.losses import SparseCategoricalCrossentropy
# from kerastuner.tuners import RandomSearch
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import h5py
import shelve
import pdb

# The paths to the h5 files
raw_train_data_path = "SynthText.h5"
raw_validation_data_path = "SynthText_val.h5"
raw_new_train_data_path = "final_sets\\train.h5"

# The path to which the modified data will be stored
train_data_path = "./train"
validation_data_path = "./validation"
new_train_data_path = "./new_train"
train_and_validation_path = "./train_and_validation"
old_train_and_new_train_path = "./old_train_and_new_train"

# Load the data
# db_train = h5py.File(train_data_path, "r")
# db_validation = h5py.File(validation_data_path, "r")
# raw_train_data = db_train["data"]
# raw_validation_data = db_validation["data"]

# Parameters
SIZE = 400


def store_data_combined(train_path, validation_path, destination_path):
    print(f"Loading {train_path}...")
    db_train = h5py.File(train_path, "r")
    raw_train = db_train["data"]
    print("Finished loading first set")

    print(f"Loading {validation_path}...")
    db_validation = h5py.File(validation_path, "r")
    raw_validation = db_validation["data"]
    print("Finished loading second set")
    counter = 0

    with shelve.open(destination_path) as shelve_db:
        print("Storing train set")
        counter = store(raw_train, counter, shelve_db)
        print("Storing validation set")
        counter = store(raw_validation, counter, shelve_db)
    print(f"Finished storing {counter} items")


def store(raw_data, counter, shelve_db):
    im_names = list(raw_data.keys())

    # Go over each image
    for i in range(len(im_names)):
        im = im_names[i]  # Go to image
        img = raw_data[im][:]  # Get the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        fonts = raw_data[im].attrs['font']  # Get the fonts
        txt = raw_data[im].attrs['txt']  # Get the text
        charBBs = raw_data[im].attrs['charBB']
        wordBBs = raw_data[im].attrs['wordBB']

        # Extract all the words from an image
        for j in range(len(txt)):
            word = txt[j].decode('utf-8')
            char_counter = 0  # Keeps track of the current char in a text

            # Extract all the characters from a word
            for k in range(len(word)):
                char = word[k]
                font = fonts[char_counter].decode('utf-8')
                charBB = charBBs[:, :, char_counter]
                char_counter += 1

                # Extract the char image
                char_image = get_char_data(img, charBB, SIZE)  # ; pdb.set_trace()

                # Reformat the image
                char_image = char_image.astype(np.float32)
                char_image /= 255.0
                char_image = char_image.reshape(list(char_image.shape) + [1])

                # Reformat the label
                label = np.array(strings_to_ints([font]))[0]

                key = f"{font}_{counter}"
                counter += 1

                shelve_db[key] = {"label": label, "char_image": char_image}
                print(f"Stored {key}")
    return counter


def store_data(dataset_path, destination_path):
    print("Loading raw data...")
    db = h5py.File(dataset_path, "r")
    raw_data = db["data"]
    print("raw data loaded")

    keys = []
    counter = 0

    with shelve.open(destination_path) as shelve_db:
        im_names = list(raw_data.keys())

        # Go over each image
        for i in range(len(im_names)):
            im = im_names[i]  # Go to image
            img = raw_data[im][:]  # Get the image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
            fonts = raw_data[im].attrs['font']  # Get the fonts
            txt = raw_data[im].attrs['txt']  # Get the text
            charBBs = raw_data[im].attrs['charBB']
            wordBBs = raw_data[im].attrs['wordBB']

            # Extract all the words from an image
            for j in range(len(txt)):
                word = txt[j].decode('utf-8');  # pdb.set_trace()
                charsData = []
                char_counter = 0  # Keeps track of the current char in a text

                # Extract all the characters from a word
                for k in range(len(word)):
                    char = word[k]
                    font = fonts[char_counter].decode('utf-8')
                    charBB = charBBs[:, :, char_counter]
                    char_counter += 1

                    # Extract the char image
                    char_image = get_char_data(img, charBB, SIZE)  # ; pdb.set_trace()

                    # Reformat the image
                    char_image = char_image.astype(np.float32)
                    char_image /= 255.0
                    char_image = char_image.reshape(list(char_image.shape) + [1])

                    # Reformat the label
                    # Reformat the label
                    label = np.array(strings_to_ints([font]))[0]

                    key = f"{font}_{counter}"
                    counter += 1

                    shelve_db[key] = {"label": label, "char_image": char_image}
                    keys.append(key)
                    print(f"Stored {key}")

    print("Finished storing modified data")
    return keys


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


def get_char_data(original_image, charBB, size):
    pts1 = np.float32([charBB[:, :].T[0], charBB[:, :].T[1], charBB[:, :].T[3], charBB[:, :].T[2]])
    pts2 = np.float32([[0, 0], [size, 0], [0, size], [size, size]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(original_image, M, (size, size))

    return result


# train_keys = store_data(raw_train_data_path, train_data_path)
# validation_keys = store_data(raw_validation_data_path, validation_data_path)
# store_data_combined(raw_train_data_path, raw_validation_data_path, train_and_validation_path)
# with shelve.open(train_and_validation_path) as shelve_db:
#     print(shelve_db["Skylark_4799"])

#validation_keys = store_data(raw_validation_data_path, validation_data_path)

#store_data_combined(raw_train_data_path, raw_new_train_data_path, old_train_and_new_train_path)




















# Diagram

model = Sequential([
  AveragePooling2D(6, 3, input_shape=(SIZE, SIZE, 1)), # pass a 6x6 grid to average the image, move the grid 3 steps each time
  Conv2D(64, 3, activation="relu"),
  Conv2D(32, 3, activation="relu"),
  MaxPool2D(2, 2), # TODO what's the difference from MaxPooling2D?
  Dropout(0.5),
  Flatten(),
  Dense(128, activation="relu"),
  Dense(3, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#print(model.summary())
