import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# Function to preprocess images
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img).astype('float32')
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def save_extracted_values_to_csv(df, extracted_values_csv_path):
    df.to_csv(extracted_values_csv_path, index=False)

def load_dataset(file_path, target_size, extracted_values_csv_path=None):
    df = pd.read_csv(file_path, decimal=",")
    images = []
    labels = []

    for _, row in df.iterrows():
        image_path = row['Foto']
        image = preprocess_image(image_path, target_size)
        images.append(image)

        npk = float(row['NpK'].replace(',', '.')) if isinstance(row['NpK'], str) else float(row['NpK'])
        ph = float(row['pH'].replace(',', '.')) if isinstance(row['pH'], str) else float(row['pH'])
        labels.append([npk, ph])

    if extracted_values_csv_path:
        save_extracted_values_to_csv(df, extracted_values_csv_path)
    
    return np.array(images), np.array(labels), df

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
    
    model = build_simplified_model(input_shape_image)
    
    file_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/dataAnalysis_augmented.csv'
    extracted_values_csv_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/extracted_values.csv'
    
    images, labels, df = load_dataset(file_path, target_size=(128, 128), extracted_values_csv_path=extracted_values_csv_path)

    X_train_img, X_test_img, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    print("Training the model...")
    model.fit(
        X_train_img, y_train,
        validation_data=(X_test_img, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[reduce_lr, early_stopping]
    )

    # Make predictions on a sample
    print("Making predictions on test sample...")
    test_index = 0
    test_image_path = df.iloc[test_index]['Foto']
    test_image = preprocess_image(test_image_path, target_size=(128, 128))
    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)
    npk, ph = prediction[0]

    # Display the actual values
    print(f"Actual NpK value: {y_test[test_index][0]}, Actual pH value: {y_test[test_index][1]}")

    print("Predicted NpK value:", npk)
    print("Predicted pH value:", ph)

    # Plot the model architecture
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    img = plt.imread('model.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
