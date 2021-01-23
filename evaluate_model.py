import numpy as np
from keras.models import load_model
from keras.utils import to_categorical, Sequence
import shelve

SIZE = 400
MODEL_PATH = "final_model_2.h5"
VALIDATION_PATH = "./validation"

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
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

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
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
          item = self.shelve_db[ID]
          #char_image = cv2.resize(item["char_image"], (SIZE, SIZE)) # TODO remove if not needed
          #char_image = char_image.reshape(list(char_image.shape) + [1])
          #backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
          #X[i,] = cv2.cvtColor(char_image, cv2.COLOR_GRAY2RGB)
          X[i,] = item["char_image"]
          y[i] = item["label"]

            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            #y[i] = self.labels[ID]

        return X, to_categorical(y, num_classes=self.n_classes)


def strings_to_ints(labels):

  ### Universal list of colors
  total_labels = ["Skylark", "Sweet Puppy", "Ubuntu Mono"]

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


# instances = count_instances(train_and_validation_data_path)
# filenames, labels = get_filenames_and_labels(train_and_validation_data_path)

instances = count_instances(VALIDATION_PATH)
filenames, labels = get_filenames_and_labels(VALIDATION_PATH)

params = {'dim': (SIZE, SIZE),
          'batch_size': 64,
          'n_classes': 3,
          'n_channels': 1,
          'shuffle': False}

loaded_model = load_model(MODEL_PATH)

with shelve.open(VALIDATION_PATH) as shelve_db:
  validation_generator = DataGenerator(shelve_db, filenames, **params)

  loaded_model.evaluate(validation_generator)
