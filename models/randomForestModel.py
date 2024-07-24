import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import export_graphviz
import graphviz

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
    
    return img_array, munsell_color_numeric[['h', 'V', 'C']].values.astype('float32')

def save_extracted_values_to_csv(munsell_colors, df, extracted_values_csv_path):
    combined_data = munsell_colors
    columns = ['h', 'V', 'C']
    df_extracted = pd.DataFrame(combined_data, columns=columns)
    
    df_final = df[['Tugas', 'Tanggal', 'Waktu ', 'Latitude', 'Longitude', 'Altitude', 'Cuaca', 'Suhu Udara', 'Precipitation', 'Kelembapan', 'NpK', 'pH', 'Foto']].reset_index(drop=True)
    
    df_final = pd.concat([df_final, df_extracted.reset_index(drop=True)], axis=1)
    df_final.to_csv(extracted_values_csv_path, index=False)

def load_dataset(file_path, munsell_file_path, target_size, extracted_values_csv_path=None):
    df = pd.read_csv(file_path, decimal=",")
    munsell_df = pd.read_csv(munsell_file_path)
    images = []
    labels = []
    munsell_colors = []

    for _, row in df.iterrows():
        image_path = row['Foto']
        _, munsell_color = preprocess_image(image_path, target_size, munsell_df)
        munsell_colors.append(munsell_color)

        npk = float(row['NpK'].replace(',', '.')) if isinstance(row['NpK'], str) else float(row['NpK'])
        ph = float(row['pH'].replace(',', '.')) if isinstance(row['pH'], str) else float(row['pH'])
        labels.append([npk, ph])

    if extracted_values_csv_path:
        munsell_colors = np.array(munsell_colors)
        save_extracted_values_to_csv(munsell_colors, df, extracted_values_csv_path)
    
    return np.array(munsell_colors), np.array(labels), df

def build_random_forest_model(input_shape):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model

def main():
    print("Starting main function...")
    
    file_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/dataAnalysis_augmented.csv'
    munsell_file_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/munsell_color_database.csv'
    extracted_values_csv_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/extracted_values.csv'
    
    features, labels, df = load_dataset(file_path, munsell_file_path, target_size=(128, 128), extracted_values_csv_path=extracted_values_csv_path)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    rf_model = build_random_forest_model(X_train.shape[1])
    print("Training the Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    print("Evaluating the model...")
    y_pred = rf_model.predict(X_test)

    results = pd.DataFrame({
        'Real NPK': y_test[:, 0],
        'Real pH': y_test[:, 1],
        'Predicted NPK': y_pred[:, 0],
        'Predicted pH': y_pred[:, 1]
    })

    print(results)

    for i, (npk, ph, pred_npk, pred_ph) in results.iterrows():
        print(f"Sample {i+1}: Real NPK: {npk:.2f}, Real pH: {ph:.2f}, Predicted NPK: {pred_npk:.2f}, Predicted pH: {pred_ph:.2f}")

    # Extracting and displaying rules from one of the trees in the Random Forest
    tree = rf_model.estimators_[0]
    export_graphviz(tree, out_file='tree.dot', feature_names=['h', 'V', 'C'], filled=True, rounded=True, special_characters=True)
    
    with open("tree.dot") as f:
        dot_graph = f.read()
    
    graphviz.Source(dot_graph).view()

if __name__ == '__main__':
    main()
