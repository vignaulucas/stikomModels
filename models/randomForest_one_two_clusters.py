import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.cluster import KMeans

# Function to find the closest Munsell color
def find_closest_munsell_color(rgb, munsell_df):
    min_dist = float('inf')
    closest_color = None
    for _, row in munsell_df.iterrows():
        munsell_color_value = np.array([row['x']])
        dist = np.linalg.norm(rgb - munsell_color_value)
        if dist < min_dist:
            min_dist = dist
            closest_color = row
    return closest_color

# Function to convert Munsell hue to a numeric code
def hue_to_numeric(hue):
    numeric_part = ''.join([char for char in hue if char.isdigit() or char == '.'])
    letter_part = ''.join([char for char in hue if char.isalpha()])
    numeric_value = float(numeric_part)
    letter_value = sum([ord(char) - ord('A') + 1 for char in letter_part])
    return numeric_value + letter_value / 100

# Function to preprocess images with one cluster
def preprocess_image_one_cluster(image_path, target_size, munsell_df):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img).astype('float32') / 255.0

    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1).fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]

    munsell_color = find_closest_munsell_color(dominant_color, munsell_df)
    munsell_color_numeric = munsell_color.copy()
    munsell_color_numeric['h'] = hue_to_numeric(munsell_color['h'])

    return img_array, munsell_color_numeric[['h', 'V', 'C']].values.astype('float32')

# Function to preprocess images with two clusters
def preprocess_image_two_clusters(image_path, target_size, munsell_df):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img).astype('float32') / 255.0

    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2).fit(pixels)
    dominant_colors = kmeans.cluster_centers_

    munsell_colors = []
    for cluster_center in dominant_colors:
        munsell_color = find_closest_munsell_color(cluster_center, munsell_df)
        munsell_color_numeric = munsell_color.copy()
        munsell_color_numeric['h'] = hue_to_numeric(munsell_color['h'])
        munsell_colors.append(munsell_color_numeric[['h', 'V', 'C']].values.astype('float32'))
    
    return img_array, np.mean(munsell_colors, axis=0)

def load_dataset(file_path, munsell_file_path, target_size):
    df = pd.read_csv(file_path, decimal=",")
    munsell_df = pd.read_csv(munsell_file_path)
    
    images = []
    labels = []
    munsell_colors_one_cluster = []
    munsell_colors_two_clusters = []

    for _, row in df.iterrows():
        image_path = row['Foto']
        
        # Process image with one cluster
        img_array_one, munsell_color_one = preprocess_image_one_cluster(image_path, target_size, munsell_df)
        munsell_colors_one_cluster.append(munsell_color_one)

        # Process image with two clusters
        img_array_two, munsell_color_two = preprocess_image_two_clusters(image_path, target_size, munsell_df)
        munsell_colors_two_clusters.append(munsell_color_two)

        images.append(img_array_one)  # Use one of the processed images for the main dataset

        npk = float(row['NpK'].replace(',', '.')) if isinstance(row['NpK'], str) else float(row['NpK'])
        ph = float(row['pH'].replace(',', '.')) if isinstance(row['pH'], str) else float(row['pH'])
        labels.append([npk, ph])
    
    return np.array(munsell_colors_one_cluster), np.array(munsell_colors_two_clusters), np.array(labels), df

def save_predictions_to_csv(predictions_one_cluster, predictions_two_clusters, file_name_one, file_name_two):
    df_one_cluster = pd.DataFrame(predictions_one_cluster, columns=['Image_Number', 'h', 'V', 'C'])
    df_two_clusters = pd.DataFrame(predictions_two_clusters, columns=['Image_Number', 'h', 'V', 'C'])

    df_one_cluster.to_csv(file_name_one, index=False)
    df_two_clusters.to_csv(file_name_two, index=False)

def main():
    file_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/dataAnalysis_augmented.csv'
    munsell_file_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/munsell_color_database.csv'
    
    features_one_cluster, features_two_clusters, labels, df = load_dataset(file_path, munsell_file_path, target_size=(128, 128))
    
    predictions_one_cluster = []
    predictions_two_clusters = []

    for i, (munsell_one, munsell_two) in enumerate(zip(features_one_cluster, features_two_clusters)):
        predictions_one_cluster.append([i+1] + list(munsell_one))
        predictions_two_clusters.append([i+1] + list(munsell_two))

    save_predictions_to_csv(predictions_one_cluster, predictions_two_clusters, 'predictions_augmented_one_cluster.csv', 'predictions_augmented_two_clusters.csv')

if __name__ == '__main__':
    main()
