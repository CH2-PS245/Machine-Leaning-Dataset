DishDash - Machine Learning Team
==
Team Member:
--
Machine Learning members of DishDash consist of:
- Yeftha Joshua Ezekiel	M006BSY0403
- Raffel Prama Andhika	M006BSY0185
- Labiba Adinda Zahwana	M200BSX0414


Importing Libraries:
--
```
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import random
```
This code imports the TensorFlow, NumPy, Pandas, Keras, Random, and Skelearn libraries, which are commonly used for machine learning and numerical computations.


Installing Libraries:
--
```
!pip install [library-name]
```
These lines use the !pip install command to install Python libraries.


Saving the Trained Model:
--
```
model.save("model.h5")
```
This line saves the trained model as an HDF5 file named "model.h5". The model was likely trained using a deep learning framework like TensorFlow.


Dataset:
--
```
- dot_prod.csv                       # Dot product result for content-based model
- final_food.csv                     # Food dataset containing nutrition and image
- food_resto_location.csv            # Dataset containing food name, restaurant, and restaurant location
- like_data.csv                      # User dataset containing user id, food id, and like
- preprocess_content_based_data.csv  # Dataset for food label
- resto_location.csv                 # Raw dataset for restaurant location
```

Model Notebook:
--
- Content-Based_Model.ipynb  --> Notebook for content based recommender system model
- Like_Based_Model.ipynb     --> Notebook for like based recommender system model


Result:
--
The model attained a training accuracy of 97.78% on the training dataset and 97.75% on the test dataset. Additionally, the model utilizes the AUC (Area Under the Receiver Operating Characteristic Curve), achieving a value of 0.937 on the training dataset and 0.933 on the test dataset.
