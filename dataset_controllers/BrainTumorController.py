import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pandas import DataFrame
from dataset_controllers.IDatasetController import IDatasetController
from PIL import Image

class BrainTumorController(IDatasetController):
    def __init__(self):
        super().__init__()
        dataset_path = r"C:\Users\furen\OneDrive\Dators\Education\School\ZPD\img"
        
        # Directories
        self.training_tumor = os.path.join(dataset_path, "training", "tumor")
        self.training_no_tumor = os.path.join(dataset_path, "training", "no_tumor")
        self.testing_tumor = os.path.join(dataset_path, "testing", "tumor")
        self.testing_no_tumor = os.path.join(dataset_path, "testing", "no_tumor")
        
        # Load and preprocess data
        self.X_train, self.y_train = self._load_data(self.training_tumor, self.training_no_tumor)
        self.X_test, self.y_test = self._load_data(self.testing_tumor, self.testing_no_tumor)
        
        # Scale data
        self.scaler = MinMaxScaler()
        self.X_train = self._scale_data(self.X_train)
        self.X_test = self._scale_data(self.X_test)
    
    def _load_data(self, tumor_dir, no_tumor_dir):
        """Load images and labels from the specified directories."""
        X, y = [], []
        
        # Load tumor images
        for file_name in os.listdir(tumor_dir):
            file_path = os.path.join(tumor_dir, file_name)
            image = self._process_image(file_path)
            if image is not None:
                X.append(image)
                y.append(1)  # Label for tumor
        
        # Load no tumor images
        for file_name in os.listdir(no_tumor_dir):
            file_path = os.path.join(no_tumor_dir, file_name)
            image = self._process_image(file_path)
            if image is not None:
                X.append(image)
                y.append(0)  # Label for no tumor
        
        return np.array(X), np.array(y)
    
    def _process_image(self, file_path):
        """Load and preprocess a single image."""
        try:
            # Open image, convert to grayscale, resize, and flatten
            image = Image.open(file_path).convert("L").resize((64, 64))  # Example size: 64x64
            return np.array(image).flatten()
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            return None
    
    def _scale_data(self, X):
        """Scale the data using MinMaxScaler."""
        return self.scaler.fit_transform(X)
    
    def get_sets(self):
        """Return the training and testing sets."""
        return self.X_train, self.X_test, self.y_train, self.y_test
