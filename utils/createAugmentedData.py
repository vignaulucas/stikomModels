import numpy as np
import pandas as pd
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, save_img
import random

# Function to load and transform an image
def transform_image(image_path, transformations):
    img = Image.open(image_path)
    for transform in transformations:
        img = transform(img)
    return img

# Function to generate a random transformation
def get_random_transformations():
    transformations = []
    # Add random rotation
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        transformations.append(lambda x: x.rotate(angle))
    # Add horizontal flip
    if random.choice([True, False]):
        transformations.append(lambda x: x.transpose(Image.FLIP_LEFT_RIGHT))
    # Add vertical flip
    if random.choice([True, False]):
        transformations.append(lambda x: x.transpose(Image.FLIP_TOP_BOTTOM))
    return transformations

# Function to create augmented dataset
def create_augmented_dataset(original_csv, output_csv, num_augmented, output_dir):
    df = pd.read_csv(original_csv, decimal=",")
    augmented_rows = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while len(augmented_rows) < num_augmented:
        for _, row in df.iterrows():
            image_path = row['Foto']
            transformations = get_random_transformations()
            transformed_image = transform_image(image_path, transformations)
            
            new_image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{len(augmented_rows)}.jpg"
            transformed_image_path = os.path.join(output_dir, new_image_name)
            transformed_image.save(transformed_image_path)
            
            new_row = row.copy()
            new_row['Foto'] = transformed_image_path
            augmented_rows.append(new_row)
            
            if len(augmented_rows) >= num_augmented:
                break
    
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df.to_csv(output_csv, index=False)

# Main function
def main():
    original_csv = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/dataAnalysis_stikom - Feuille 1.csv'
    output_csv = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/dataAnalysis_augmented.csv'
    output_dir = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/augmented_images'
    num_augmented = 500
    
    create_augmented_dataset(original_csv, output_csv, num_augmented, output_dir)
    print(f"Augmented dataset saved to {output_csv}")

if __name__ == '__main__':
    main()
