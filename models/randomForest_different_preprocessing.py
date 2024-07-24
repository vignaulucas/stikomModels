import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error
import graphviz
import os
from PIL import Image
import cv2

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
    numeric_part = ''.join([char for char in hue if char.isdigit() or char == '.'])
    letter_part = ''.join([char for char in hue if char.isalpha()])
    numeric_value = float(numeric_part)
    letter_value = sum([ord(char) - ord('A') + 1 for char in letter_part])
    return numeric_value + letter_value / 100

# Function to remove unwanted elements from the image
def remove_unwanted_elements(image_path, cleaned_path):
    # Open image using PIL
    picture = Image.open(image_path).convert('RGB')
    img_array = np.array(picture)
    
    # Convert RGB to BGR for OpenCV
    img_array = img_array[:, :, ::-1]
    
    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    
    # Define the color range for green elements (e.g., grass)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Define the color range for black elements
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    
    # Create masks for green and black colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Combine masks
    mask = cv2.bitwise_or(mask_green, mask_black)
    
    # Invert mask to keep the soil
    mask_inv = cv2.bitwise_not(mask)
    
    # Apply the mask to the original image
    cleaned_img_array = cv2.bitwise_and(img_array, img_array, mask=mask_inv)
    
    # Convert BGR to RGB for PIL
    cleaned_img_array = cleaned_img_array[:, :, ::-1]
    
    # Save cleaned image
    cleaned_img = Image.fromarray(cleaned_img_array)
    if not os.path.exists(cleaned_path):
        os.makedirs(cleaned_path)
    cleaned_image_path = os.path.join(cleaned_path, os.path.basename(image_path))
    cleaned_img.save(cleaned_image_path)
    
    # Calculate the dominant color in the cleaned image (excluding black areas)
    non_black_pixels = cleaned_img_array[mask_inv != 0]
    if len(non_black_pixels) > 0:
        # Calculate the mean color of the non-black pixels
        dominant_color = np.mean(non_black_pixels, axis=0).astype(int)
    else:
        dominant_color = np.array([128, 128, 128])  # Default to gray if no non-black pixels found
    
    # Save the segmented image as a solid color square
    segmented_img_array = np.full_like(cleaned_img_array, dominant_color)
    
    return segmented_img_array

# Function to preprocess images and extract Munsell color features
def preprocess_image(image_path, target_size, munsell_df, cleaned_path, segmented_path):
    cleaned_img_array = remove_unwanted_elements(image_path, cleaned_path)
    img = Image.fromarray(cleaned_img_array[:, :, ::-1])  # Convert BGR to RGB for PIL
    img = img.resize(target_size)
    img_array = img_to_array(img).astype('float32')
    img_array = img_array / 255.0  # Normalize to [0, 1]

    # Save the segmented image
    segmented_img = Image.fromarray((img_array * 255).astype(np.uint8))
    if not os.path.exists(segmented_path):
        os.makedirs(segmented_path)
    segmented_image_path = os.path.join(segmented_path, os.path.basename(image_path))
    segmented_img.save(segmented_image_path)
    
    # Extract dominant color using KMeans clustering
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1).fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    
    # Find the closest Munsell color
    munsell_color = find_closest_munsell_color(dominant_color, munsell_df)
    
    # Convert hue to numeric
    munsell_color_numeric = munsell_color.copy()
    munsell_color_numeric['h'] = hue_to_numeric(munsell_color['h'])
    
    return img_array, munsell_color_numeric[['h', 'V', 'C']].values.astype('float32')

def load_dataset(file_path, munsell_file_path, target_size, cleaned_path, segmented_path):
    df = pd.read_csv(file_path, decimal=",")
    munsell_df = pd.read_csv(munsell_file_path)
    images = []
    labels = []
    munsell_colors = []

    for _, row in df.iterrows():
        image_path = row['Foto']
        _, munsell_color = preprocess_image(image_path, target_size, munsell_df, cleaned_path, segmented_path)
        munsell_colors.append(munsell_color)

        npk = float(row['NpK'].replace(',', '.')) if isinstance(row['NpK'], str) else float(row['NpK'])
        ph = float(row['pH'].replace(',', '.')) if isinstance(row['pH'], str) else float(row['pH'])
        labels.append([npk, ph])
    
    return np.array(munsell_colors), np.array(labels), df

def build_random_forest_model(input_shape):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model

def save_predictions_to_csv(predictions, file_name):
    df = pd.DataFrame(predictions, columns=['Image_Number', 'h', 'V', 'C'])
    df.to_csv(file_name, index=False)

def main():
    print("Starting main function...")
    
    file_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/dataAnalysis_augmented.csv'
    munsell_file_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/munsell_color_database.csv'
    cleaned_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/preprocessing_steps_augmentedData/cleaned'
    segmented_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/preprocessing_steps_augmentedData/segmented'
    target_size = (128, 128)
    
    features, labels, df = load_dataset(file_path, munsell_file_path, target_size, cleaned_path, segmented_path)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    rf_model = build_random_forest_model(X_train.shape[1])
    print("Training the Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    print("Evaluating the model...")
    y_pred = rf_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    
    results = pd.DataFrame({
        'Real NPK': y_test[:, 0],
        'Real pH': y_test[:, 1],
        'Predicted NPK': y_pred[:, 0],
        'Predicted pH': y_pred[:, 1]
    })

    print("Results:")
    print(results)

    # Extracting and displaying rules from one of the trees in the Random Forest
    tree = rf_model.estimators_[0]
    export_graphviz(tree, out_file='tree.dot', feature_names=['h', 'V', 'C'], filled=True, rounded=True, special_characters=True)
    
    with open("tree.dot") as f:
        dot_graph = f.read()
    
    graphviz.Source(dot_graph).view()

if __name__ == '__main__':
    main()
