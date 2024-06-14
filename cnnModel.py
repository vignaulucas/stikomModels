import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, concatenate
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
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


# Function to build the multimodal neural network model
def build_multimodal_model(input_shape_image, input_shape_weather, input_shape_munsell):
    cnn_input = Input(shape=input_shape_image)
    x = Conv2D(32, (3, 3), activation='relu')(cnn_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    
    weather_input = Input(shape=(input_shape_weather,))
    y = Dense(128, activation='relu')(weather_input)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    
    munsell_input = Input(shape=(input_shape_munsell,))
    z = Dense(32, activation='relu')(munsell_input)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)

    combined = concatenate([x, y, z])
    a = Dense(512, activation='relu')(combined)
    a = Dropout(0.5)(a)
    a = Dense(256, activation='relu')(a)
    a = Dropout(0.4)(a)
    a = Dense(128, activation='relu')(a)
    a = Dropout(0.3)(a)
    output = Dense(2, activation='softplus')(a)  # Predict pH and NPK levels
    
    model = Model(inputs=[cnn_input, weather_input, munsell_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    return model

# Main function
def main():
    print("Starting main function...")
    input_shape_image = (128, 128, 3)
    input_shape_weather = 3  # Suhu Udara, Precipitation, Kelembapan
    input_shape_munsell = 3  # Hue, Value, Chroma
    
    model = build_multimodal_model(input_shape_image, input_shape_weather, input_shape_munsell)
    
    file_path = '/Users/vignaulucas/Desktop/stikom_models/dataAnalysis_stikom - Feuille 1.csv'
    munsell_file_path = '/Users/vignaulucas/Desktop/stikom_models/munsell_color_database.csv'  # Update with the actual path to the Munsell database file
    images, labels, weather_data, munsell_features, scaler, df = load_dataset(file_path, munsell_file_path, target_size=(128, 128))

    
    X_train_img, X_test_img, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train_weather, X_test_weather = train_test_split(weather_data, test_size=0.2, random_state=42)
    X_train_munsell, X_test_munsell = train_test_split(munsell_features, test_size=0.2, random_state=42)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    
    print("Training the model...")
    history = model.fit(
        [X_train_img, X_train_weather, X_train_munsell], y_train,
        validation_data=([X_test_img, X_test_weather, X_test_munsell], y_test),
        epochs=50,
        batch_size=10,
        callbacks=[reduce_lr, early_stopping]
    )

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Make predictions on a sample
    print("Making predictions on test sample...")
    test_index = 0
    test_image_path = df.iloc[test_index]['Foto']
    test_image, _ = preprocess_image(test_image_path, target_size=(128, 128), munsell_df=pd.read_csv(munsell_file_path))
    test_image = np.expand_dims(test_image, axis=0)

    test_weather = np.array([[df.iloc[test_index]['Suhu Udara'], df.iloc[test_index]['Precipitation'], df.iloc[test_index]['Kelembapan']]])
    test_weather = scaler.transform(test_weather)
    test_munsell = np.expand_dims(munsell_features[test_index], axis=0)

    prediction = model.predict([test_image, test_weather, test_munsell])
    npk, ph = prediction[0]
    print("Predicted NpK value:", npk)
    print("Predicted pH value:", ph)

if __name__ == '__main__':
    main()


