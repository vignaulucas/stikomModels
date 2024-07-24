STIKOM Soil Analysis and Prediction Models
This repository contains the code and data for my internship project at STIKOM Bali, focusing on predicting soil NPK and pH levels from soil images. The project involves multiple machine learning models, including Convolutional Neural Networks (CNN), Gradient Boosting, and Random Forest. Various preprocessing techniques have been applied to improve model accuracy.

Project Overview
Initial Approach
At the beginning of my internship, I developed a highly complex model with numerous parameters, hoping to achieve accurate predictions of soil pH and NPK concentrations from soil images. Unfortunately, this initial model did not perform well, as it was prone to overfitting and struggled with the limited dataset available. Consequently, I decided to adopt a different approach by simplifying the model architecture, which ultimately proved to be more precise and effective.

Model Descriptions
Convolutional Neural Network (CNN)
The CNN model was one of the initial approaches. It included several convolutional and pooling layers followed by dense layers to predict the NPK and pH values from soil images.

Gradient Boosting
This model has always been precise and its accuracy has further improved since increasing the dataset to 500 samples.

Random Forest Models
Original Random Forest Model

The Random Forest model has consistently been the most accurate among the three models since the beginning.

Improved Random Forest Model with Two Clusters

I then tried to create the same model but with two clusters instead of one. This allowed me to obtain a different preprocessing. The accuracy of this model was better than the first, improving the Random Forest Mean Squared Error (MSE) from 0.1429 for 30 data samples to 0.1327.

Final Version of the Random Forest Model

The final and most precise version of the Random Forest model was obtained by changing the preprocessing method. This involved adding filters to remove green and black elements and aiming to obtain the dominant color of the soil while excluding unwanted elements in the images.

Preprocessing Images
The preprocessing images for this version are available in the following folders:

preprocessing_steps_30_datas: Contains subfolders for the 30 data points.
cleaned: First version where only green objects like grass were removed.
cleanedV2: Final version.
segmented_unwanted_element: Segmented version for unwanted elements.
segmentedV2: Improved segmented version.
preprocessing_steps_augmentedData: Contains similar subfolders for the 500 data points.
Generating Prediction CSV Files
I generated CSV files containing my predictions to obtain the h, V, and C values corresponding to the Munsell color system for both one and two clusters. These predictions were made for datasets containing 30 and 500 data points. The structure of these files includes columns for the image number and the predicted h, V, and C values.

These prediction CSV files are available in the predictions folder:

predictions_30_datas: Contains prediction files for the dataset with 30 data points.
predictions_500_datas: Contains prediction files for the dataset with 500 data points.
Each of these folders further includes files for one and two clusters, named accordingly for clarity.
Comparison of Models

Finally, I created a comparison model to evaluate the performance of the three original models: CNN, Gradient Boosting, and Random Forest. Here are the results obtained from this comparison:

Precision for 30 datas:

Gradient Boosting NPK MSE: 0.3681
Gradient Boosting pH MSE: 0.1731
Random Forest MSE: 0.1429
CNN Model MSE: 0.4893

Precision for 500 datas:

Gradient Boosting NPK MSE: 0.0335
Gradient Boosting pH MSE: 0.0084
Random Forest MSE:  0.0183
CNN Model MSE: 0.0363


This comparison helped in understanding the strengths and weaknesses of each model and confirmed that the Random Forest model, particularly with the final preprocessing improvements, provided the best accuracy for predicting soil NPK and pH values.
