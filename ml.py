from tensorflow import keras
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import cv2
from glob import glob
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

loaded_model = keras.models.load_model('results')
path='uploadbyuser'

dataset_path = os.listdir(path)
im_size = 224

images = []
labels = []


for f in glob(path+'/*'):
   
    img = cv2.imread(f)

    img = cv2.resize(img, (224, 224))
    images.append(img)
images = np.array(images)
images = images.astype('float32') / 255.0
predictions=loaded_model.predict(images)
for i, pred_probs in enumerate(predictions):

    class_labels = ["Class A", "Class B"]
    predicted_class = class_labels[np.argmax(pred_probs)]
    print(f"Sample {i + 1} - Predicted: {predicted_class}, Probabilities: {pred_probs}")