import numpy as np
import sys
import h5py
import csv
import cv2
from keras.models import load_model
from keras.utils import to_categorical
from matplotlib import pyplot as plt


# Constants
SIZE = 400
RESULTS_PATH = "test_predictions.csv"


def main():
    # Read the model and test set paths from the command line arguments
    if len(sys.argv) != 3:
        raise ValueError("Model path and Test Set paths are required")
    model_path = sys.argv[1]
    test_set_path = sys.argv[2]

    # Evaluate the model and write the results into a file
    evaluate_model(model_path, test_set_path)


def evaluate_model(model_path, test_set_path):
    """
    Evaluates the given model using the given test set and writes the results into a file.

    :param model_path: path to the model containing all the needed info to use it
    :param test_set_path: path to the test set
    """
    try:
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded")
    except Exception:
        raise ValueError("Failed to load model")
    try:
        print("Loading test set...")
        # image_names, chars_text, chars_images = load_test_set(test_set_path)
        db = h5py.File(test_set_path, "r")
        raw_data = db["data"]
        print("Test set loaded")
    except Exception:
        raise ValueError("Failed to load test set")

    # Predict the fonts using the model
    print("Predicting fonts...")
    # predictions = model.predict(chars_images)
    image_names, chars_text, predictions = predict_on_data(raw_data, model)
    print("Finished predicting")

    try:
        print("Writing results to csv...")
        print_results_to_csv(image_names, chars_text, predictions, RESULTS_PATH)
        print("Finished writing results, file name: {0}".format(RESULTS_PATH))
    except Exception:
        raise ValueError("Failed writing results to csv")


def predict_on_data(raw_data, model):
    """
    Performs a prediction on the given raw data by first extracting the chars from the images,
    then pre-processing them and finally feeding them to the given model.

    :param raw_data: data to predict on
    :param model: used to predict on the data
    :return: list of image names, list of the chars on which the prediction was made and list of the predictions
    in the form of a one-hot vector
    """
    im_names = list(raw_data.keys())
    image_names = []
    chars_text = []
    predictions = []
    counter = 0 # TODO delete

    # Go over each image
    for i in range(len(im_names)):

        counter += 1
        if counter == 2:
            break

        im = im_names[i]  # Go to image
        img = raw_data[im][:]  # Get the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        #fonts = raw_data[im].attrs['font']  # Get the fonts
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
                # font = fonts[char_counter].decode('utf-8')
                charBB = charBBs[:, :, char_counter]
                char_counter += 1

                # Extract the char image
                char_image = extract_char_image(img, charBB, SIZE)

                # Reformat the image
                char_image = char_image.astype(np.float32)
                char_image /= 255.0
                char_image = char_image.reshape(list(char_image.shape) + [1])
                plt.imshow(char_image, cmap="gray")
                plt.imshow(char_image, cmap="gray")

                print(f"Predicting on char: {char} in image: {im}")
                prediction = model.predict_classes(np.array([char_image, ]))[0]
                prediction = to_categorical(prediction, num_classes=3)

                image_names.append(im)
                chars_text.append(char)
                predictions.append(prediction)

    return np.array(image_names), np.array(chars_text), np.array(predictions)


def extract_char_image(original_image, charBB, size):
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


def print_results_to_csv(image_names, chars_text, predictions, results_file_path):
    """
    Prints the prediction results into a csv file.

    :param image_names: names of images
    :param chars_text: chars values
    :param predictions: prediction for each char's font
    :param results_file_path: path of the result csv file
    """
    with open(results_file_path, mode='w', newline='') as results_csv:
        fieldnames = ["", "image", "char", b'Skylark', b'Sweet Puppy', b'Ubuntu Mono']
        writer = csv.DictWriter(results_csv, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(chars_text)):
            prediction = predictions[i]
            row_data = {
                "": i,
                "image": image_names[i],
                "char": chars_text[i],
                b'Skylark': prediction[1],
                b'Sweet Puppy': prediction[2],
                b'Ubuntu Mono': prediction[0]
            }
            writer.writerow(row_data)


if __name__ == "__main__":
    main()
