import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, Sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, AveragePooling2D, Dropout
import h5py
import shelve
import sys

# Constants
TRAIN_AND_VALIDATION_SET_PATH = "./train_and_validation_set"
SIZE = 400


def main():
    # Read the train and validation sets paths from the command line arguments
    if len(sys.argv) != 3:
        raise ValueError("Training and validation set paths are required")
    raw_training_set_path = sys.argv[1]
    raw_validation_set_path = sys.argv[2]

    # Extract all the char images from the given data sets and save them as a combined new data set
    store_data_combined(raw_training_set_path, raw_validation_set_path, TRAIN_AND_VALIDATION_SET_PATH)


def store_data_combined(train_path, validation_path, destination_path):
    """
    Extracts all the char images from the given training and validation sets, performs a pre-process
    on the images and stores them in a combined set.
    :param train_path: path to .h5 training set
    :param validation_path: path to .h5 validation set
    :param destination_path: path of the destination combined set
    """
    print(f"Loading {train_path}...")
    db_train = h5py.File(train_path, "r")
    raw_train = db_train["data"]
    print("Finished loading training set")

    print(f"Loading {validation_path}...")
    db_validation = h5py.File(validation_path, "r")
    raw_validation = db_validation["data"]
    print("Finished loading validation set")
    counter = 0

    with shelve.open(destination_path) as shelve_db:
        print("Storing train set")
        counter = store(raw_train, counter, shelve_db)
        print("Storing validation set")
        counter = store(raw_validation, counter, shelve_db)
    print(f"Finished storing {counter} items")


def store(raw_data, counter, shelve_db):
    """
    Extracts char images from the given data and stores them in the given shelve db.
    :param raw_data: data set to extract images from
    :param counter: used for the naming of the keys
    :param shelve_db: shelve db
    :return: the number of the last image on which the counter stopped
    """
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

                # Store the char image and the label
                shelve_db[key] = {"label": label, "char_image": char_image}
                print(f"Stored {key}")
    return counter


def strings_to_ints(labels):
    """
    Converts the given labels from strings to integers.
    :param labels: list of string
    :return: list of integers
    """
    # Universal list of colors
    total_labels = ["Skylark", "Sweet Puppy", "Ubuntu Mono"]

    # map each color to an integer
    mapping = {}
    for x in range(len(total_labels)):
        mapping[total_labels[x]] = x

    # integer representation
    for x in range(len(labels)):
        labels[x] = mapping[labels[x]]

    return labels


def get_char_data(original_image, charBB, size):
    """
    Extracts a single char from the given image using the given char bounding box.

    :param original_image: image containing the char
    :param charBB: char bounding box
    :param size: desired result image size
    :return: image of the char in the original image
    """
    # Define the source and destination coordinates
    source_coordinates = np.float32([charBB[:, :].T[0], charBB[:, :].T[1],
                                     charBB[:, :].T[3], charBB[:, :].T[2]])
    destination_coordinates = np.float32([[0, 0], [size, 0],
                                          [0, size], [size, size]])

    # Perform perspective transformation
    transform = cv2.getPerspectiveTransform(source_coordinates, destination_coordinates)

    # Transform the original image using the given transform matrix
    result = cv2.warpPerspective(original_image, transform, (size, size))

    return result


class DataGenerator(Sequence):
    "Used to generate data for a keras model"

    def __init__(self, shelve_db, list_IDs, batch_size=32, dim=(SIZE, SIZE), n_channels=1,
                 n_classes=3, shuffle=True):

        # The storage from which the data is read
        self.shelve_db = shelve_db

        # The image dimension
        self.dim = dim

        # Size of the batch that's fed to the model
        self.batch_size = batch_size

        # List of IDs that act as keys when retrieving data from the storage
        self.list_IDs = list_IDs

        # Number of input image channels
        self.n_channels = n_channels

        # Number of classes from which the data consists
        self.n_classes = n_classes

        # Should the data be shuffled
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            item = self.shelve_db[ID]
            X[i,] = add_noise(item["char_image"])
            y[i] = item["label"]

        return X, to_categorical(y, num_classes=self.n_classes)


def add_noise(image):
    """
    Adds random noise to the given image.
    :param image: image to add noise to
    :return: image with random noise
    """
    row, col, channel = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, channel))
    gauss = np.clip(gauss, 0, 255)
    gauss = gauss.reshape(row, col, channel)
    noisy_image = image + gauss

    return noisy_image


def get_filenames_and_labels(path):
    """
    Retrieves the names and corresponding labels of all the files in the given path
    :param path: path to storage of files and labels
    :return: list of file names, list of labels
    """
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


def train(training_set_path, model_destination_path):
    """
    Trains a CNN model using the given training set and stores it in the given destination path.
    :param training_set_path: path to training set
    :param model_destination_path: path in which the trained model will be stored
    """
    filenames, labels = get_filenames_and_labels(training_set_path)

    # Split the data into train and test sets
    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(filenames, labels,
                                                                          train_size=0.8,
                                                                          random_state=42,
                                                                          stratify=labels)

    # Define the model architecture
    model = Sequential([
        AveragePooling2D(6, 3, input_shape=(SIZE, SIZE, 1)),
        # pass a 6x6 grid to average the image, move the grid 3 steps each time
        Conv2D(64, 3, activation="relu"),
        Conv2D(32, 3, activation="relu"),
        MaxPool2D(2, 2),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(3, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Define the generator parameters
    params = {'dim': (SIZE, SIZE),
              'batch_size': 64,
              'n_classes': 3,
              'n_channels': 1,
              'shuffle': True}

    # Read from the file of the new training set and feed the data to the model using the generators
    with shelve.open(training_set_path) as shelve_db:

        # Define a training and validation generators
        training_generator = DataGenerator(shelve_db, X_train_filenames, **params)
        validation_generator = DataGenerator(shelve_db, X_val_filenames, **params)

        # Train the model
        model.fit(training_generator,
                  epochs=15,
                  validation_data=validation_generator,
                  use_multiprocessing=True,
                  workers=6)

        # Save the model for future use
        model.save(model_destination_path)


if __name__ == "__main__":
    main()
