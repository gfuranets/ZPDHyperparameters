from sklearn.preprocessing import MinMaxScaler
import os
import cv2
import numpy as np
from pandas import DataFrame
from dataset_controllers.IDatasetController import IDatasetController

# Function to load images from a folder
def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

# Function to resize images
def preprocess_images(images, target_size=(64, 64)):
    resized_images = []
    for img in images:
        img_resized = cv2.resize(img, target_size)  # Resize to 64x64
        resized_images.append(img_resized)
    return np.array(resized_images)

class BrainTumorDatasetController(IDatasetController):
    def __init__(self):
        # Define paths for the dataset
        testing_tumor_folder = r"C:\Users\furen\OneDrive\Dators\Education\School\ZPD\img\testing\tumor"
        testing_no_tumor_folder = r"C:\Users\furen\OneDrive\Dators\Education\School\ZPD\img\testing\no_tumor"
        training_tumor_folder = r"C:\Users\furen\OneDrive\Dators\Education\School\ZPD\img\training\tumor"
        training_no_tumor_folder = r"C:\Users\furen\OneDrive\Dators\Education\School\ZPD\img\training\no_tumor"
        
        # Load images and their labels
        self.train_tumor_images, self.train_tumor_labels = load_images_from_folder(training_tumor_folder, label=1)
        self.train_no_tumor_images, self.train_no_tumor_labels = load_images_from_folder(training_no_tumor_folder, label=0)
        self.test_tumor_images, self.test_tumor_labels = load_images_from_folder(testing_tumor_folder, label=1)
        self.test_no_tumor_images, self.test_no_tumor_labels = load_images_from_folder(testing_no_tumor_folder, label=0)

        # Preprocess images (resize them)
        self.train_tumor_images = preprocess_images(self.train_tumor_images)
        self.train_no_tumor_images = preprocess_images(self.train_no_tumor_images)
        self.test_tumor_images = preprocess_images(self.test_tumor_images)
        self.test_no_tumor_images = preprocess_images(self.test_no_tumor_images)

        # Combine all images for easier handling
        self.images = (self.train_tumor_images,
                       self.train_no_tumor_images,
                       self.test_tumor_images,
                       self.test_no_tumor_images)
        
    def get_sets(self):
        # Combine images and labels into training and testing sets
        X_train = np.concatenate([self.train_tumor_images, self.train_no_tumor_images], axis=0)
        y_train = np.concatenate([self.train_tumor_labels, self.train_no_tumor_labels], axis=0)

        X_test = np.concatenate([self.test_tumor_images, self.test_no_tumor_images], axis=0)
        y_test = np.concatenate([self.test_tumor_labels, self.test_no_tumor_labels], axis=0)

        # Normalize pixel values to the range [0, 1]
        X_train /= 255.0
        X_test /= 255.0

        return X_train, X_test, y_train, y_test
