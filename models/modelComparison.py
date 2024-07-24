from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
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

# Function to preprocess images and extract Munsell color features
def preprocess_image(image_path, target_size, munsell_df):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img).astype('float32')
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1).fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    
    munsell_color = find_closest_munsell_color(dominant_color, munsell_df)
    
    munsell_color_numeric = munsell_color.copy()
    munsell_color_numeric['h'] = hue_to_numeric(munsell_color['h'])
    
    return img_array, munsell_color_numeric[['h', 'V', 'C']].values.astype('float32')

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

    return np.array(images), np.array(labels), np.array(munsell_colors), df

def build_gradient_boosting_model():
    model_npk = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, loss='ls')
    model_ph = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, loss='ls')
    return model_npk, model_ph

def build_random_forest_model():
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    return rf_model

# Function to build the simplified neural network model
def build_simplified_model(input_shape_image):
    cnn_input = Input(shape=input_shape_image)
    x = Conv2D(32, (3, 3), activation='relu')(cnn_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    
    x = Dense(64, activation='relu')(x)
    output = Dense(2)(x)  # Activation linéaire pour prédire pH et NPK
    
    model = Model(inputs=cnn_input, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    return model

def main():
    print("Starting main function...")
    input_shape_image = (128, 128, 3)
    
    gb_model_npk, gb_model_ph = build_gradient_boosting_model()
    rf_model = build_random_forest_model()
    cnn_model = build_simplified_model(input_shape_image)
    
    file_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/dataAnalysis_augmented.csv'
    munsell_file_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/munsell_color_database.csv'
    images, labels, munsell_features, df = load_dataset(file_path, munsell_file_path, target_size=(128, 128))

    # Flatten image data for the models
    flattened_images = images.reshape(images.shape[0], -1)
    combined_features = np.hstack((flattened_images, munsell_features))
    
    X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)
    
    # Split labels for NPK and pH for gradient boosting model
    y_train_npk = y_train[:, 0]
    y_train_ph = y_train[:, 1]
    y_test_npk = y_test[:, 0]
    y_test_ph = y_test[:, 1]

    print("Training the Gradient Boosting NPK model...")
    gb_model_npk.fit(X_train, y_train_npk)
    
    print("Training the Gradient Boosting pH model...")
    gb_model_ph.fit(X_train, y_train_ph)
    
    print("Training the Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # Train the CNN model
    X_train_img, X_test_img, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    print("Training the CNN model...")
    history = cnn_model.fit(
        X_train_img, y_train,
        validation_data=(X_test_img, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[reduce_lr, early_stopping]
    )

    print("Making predictions on test sample...")
    predictions_npk_gb = gb_model_npk.predict(X_test)
    predictions_ph_gb = gb_model_ph.predict(X_test)
    predictions_rf = rf_model.predict(X_test)
    
    test_predictions_cnn = cnn_model.predict(X_test_img)

    # Calculate Mean Squared Error for both models
    mse_npk_gb = mean_squared_error(y_test_npk, predictions_npk_gb)
    mse_ph_gb = mean_squared_error(y_test_ph, predictions_ph_gb)
    mse_rf = mean_squared_error(y_test, predictions_rf)
    mse_cnn = mean_squared_error(y_test, test_predictions_cnn)
    
    print(f"Gradient Boosting NPK MSE: {mse_npk_gb:.4f}")
    print(f"Gradient Boosting pH MSE: {mse_ph_gb:.4f}")
    print(f"Random Forest MSE: {mse_rf:.4f}")
    print(f"CNN Model MSE: {mse_cnn:.4f}")
    
    if mse_npk_gb + mse_ph_gb < mse_rf and mse_npk_gb + mse_ph_gb < mse_cnn:
        print("The Gradient Boosting model is more accurate.")
    elif mse_rf < mse_npk_gb + mse_ph_gb and mse_rf < mse_cnn:
        print("The Random Forest model is more accurate.")
    else:
        print("The CNN model is more accurate.")
    
    # Detailed prediction outputs
    for idx in range(len(predictions_rf)):
        print(f"Sample {idx + 1}:")
        print(f"  Gradient Boosting - Predicted NPK: {predictions_npk_gb[idx]:.2f}, Actual NPK: {y_test_npk[idx]:.2f}")
        print(f"  Gradient Boosting - Predicted pH: {predictions_ph_gb[idx]:.2f}, Actual pH: {y_test_ph[idx]:.2f}")
        print(f"  Random Forest - Predicted NPK: {predictions_rf[idx][0]:.2f}, Predicted pH: {predictions_rf[idx][1]:.2f}")
        print(f"  CNN - Predicted NPK: {test_predictions_cnn[idx][0]:.2f}, Predicted pH: {test_predictions_cnn[idx][1]:.2f}")
        print(f"  Actual NPK: {y_test[idx][0]:.2f}, Actual pH: {y_test[idx][1]:.2f}")
    
if __name__ == '__main__':
    main()
