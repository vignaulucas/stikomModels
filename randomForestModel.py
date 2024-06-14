import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, concatenate
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Function to find the closest Munsell color
def find_closest_munsell_color(rgb, munsell_df):
    min_dist = float('inf')
    closest_color = None
    for _, row in munsell_df.iterrows():
        munsell_color_value = np.array([row['x']])  # Adjust this if 'x' is a composite value
        dist = np.linalg.norm(rgb - munsell_color_value)
        if dist < min_dist:
            min_dist = dist
            closest_color = row
    return closest_color

# Function to convert Munsell hue to a numeric code
def hue_to_numeric(hue):
    # Extract the numeric part and the letter part separately
    numeric_part = ''.join([char for char in hue if char.isdigit() or char == '.'])
    letter_part = ''.join([char for char in hue if char.isalpha()])
    # Convert the numeric part to a float
    numeric_value = float(numeric_part)
    # Assign a numeric value to the letter part
    letter_value = sum([ord(char) - ord('A') + 1 for char in letter_part])
    # Combine both parts to form a unique numeric value
    return numeric_value + letter_value / 100

# Function to preprocess images and extract Munsell color features
def preprocess_image(image_path, target_size, munsell_df):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img).astype('float32')
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Extract dominant color using KMeans clustering
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1).fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    
    # Find the closest Munsell color
    munsell_color = find_closest_munsell_color(dominant_color, munsell_df)
    
    # Convert hue to numeric
    munsell_color_numeric = munsell_color.copy()
    munsell_color_numeric['h'] = hue_to_numeric(munsell_color['h'])
    
    # Return image array and Munsell color features
    return img_array, munsell_color_numeric[['h', 'V', 'C']].values.astype('float32')

# Function to preprocess and align weather data
def preprocess_weather_data(df):
    df['Suhu Udara'] = df['Suhu Udara'].str.replace('°C', '').astype(float)
    df['Precipitation'] = df['Precipitation'].str.replace('%', '').astype(float)
    df['Kelembapan'] = df['Kelembapan'].str.replace('%', '').astype(float)
    
    weather_data = df[['Suhu Udara', 'Precipitation', 'Kelembapan']].copy()
    
    scaler = StandardScaler()
    weather_data = scaler.fit_transform(weather_data).astype('float32')
    
    return weather_data, scaler

def load_dataset(file_path, munsell_file_path, target_size):
    df = pd.read_csv(file_path, decimal=",")
    munsell_df = pd.read_csv(munsell_file_path)
    images = []
    labels = []
    munsell_colors = []

    for _, row in df.iterrows():
        image_path = row['Foto']
        image, munsell_color = preprocess_image(image_path, target_size, munsell_df)
        images.append(image)
        munsell_colors.append(munsell_color)

        npk = float(row['NpK'].replace(',', '.')) if isinstance(row['NpK'], str) else float(row['NpK'])
        ph = float(row['pH'].replace(',', '.')) if isinstance(row['pH'], str) else float(row['pH'])
        labels.append([npk, ph])

    weather_data, scaler = preprocess_weather_data(df)
    return np.array(images), np.array(labels), np.array(weather_data), np.array(munsell_colors), scaler, df


# Function to build the Random Forest model
def build_random_forest_model(input_shape_image, input_shape_weather, input_shape_munsell):
    # Assuming that you have a feature extraction or reduction method that results in a 1D vector
    flattened_image_features = input_shape_image  # This should be the number of features after flattening or feature extraction
    weather_features = input_shape_weather
    munsell_features = input_shape_munsell

    # Total features is the sum of features from all modalities
    total_features = flattened_image_features + weather_features + munsell_features

    # Create Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    return rf_model


def main():
    print("Starting main function...")
    input_shape_image = 49152  # For example, if images are 128x128x3 and you flatten them
    input_shape_weather = 3  # Suhu Udara, Precipitation, Kelembapan
    input_shape_munsell = 3  # Hue, Value, Chroma
    
    rf_model = build_random_forest_model(input_shape_image, input_shape_weather, input_shape_munsell)
    
    # Load and prepare your dataset
    file_path = '/Users/vignaulucas/Desktop/stikom_models/dataAnalysis_stikom - Feuille 1.csv'
    munsell_file_path = '/Users/vignaulucas/Desktop/stikom_models/munsell_color_database.csv'
    images, labels, weather_data, munsell_features, scaler, df = load_dataset(file_path, munsell_file_path, target_size=(128, 128))
    
    # Flatten images and concatenate all features
    images_flattened = images.reshape(images.shape[0], -1)
    features = np.concatenate([images_flattened, weather_data, munsell_features], axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    print("Training the Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating the model...")
    y_pred = rf_model.predict(X_test)
    for i, (npk, ph) in enumerate(y_pred):
        print(f"Sample {i+1}: Predicted NPK concentration: {npk:.2f}, Predicted pH: {ph:.2f}")
    
    # Testing with a new image
    test_image_path = '/Users/vignaulucas/Desktop/stikom_models/tnh 1.jpg'
    test_image, test_munsell = preprocess_image(test_image_path, target_size=(128, 128), munsell_df=pd.read_csv(munsell_file_path))
    test_image_flattened = test_image.reshape(1, -1)
    test_features = np.concatenate([test_image_flattened, weather_data[0].reshape(1, -1), munsell_features[0].reshape(1, -1)], axis=1)
    
    test_prediction = rf_model.predict(test_features)
    test_npk, test_ph = test_prediction[0]
    print(f"Test Image Prediction: NPK concentration: {test_npk:.2f}, pH: {test_ph:.2f}")

if __name__ == '__main__':
    main()
