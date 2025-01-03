# Facial Emotion Recognition using Convolutional Neural Networks (CNN)
## Project Overview
This project focuses on developing a Convolutional Neural Network (CNN) for recognizing facial emotions from images. The CNN model is trained using the CK+ (Extended Cohn-Kanade) dataset, which contains facial expression images labeled with corresponding emotions.

## Project Status
The CNN model implementation for facial emotion recognition has been completed.
The model was trained and evaluated on the CK+ dataset.

## Dataset
The project uses the CK+ (Extended Cohn-Kanade) dataset, which consists of images showing various facial expressions labeled with their corresponding emotions. The dataset is publicly available on Kaggle.

## System Description

### 1. Data Preprocessing
Loaded the CK+ dataset and explored its basic information.
Converted pixel strings from the dataset into NumPy arrays.
Resized images to a standard size (48x48 pixels).
Normalized pixel values to a range of [0, 1] for better model performance.
Dynamically determined the number of emotion classes in the dataset.
Split the dataset into features (images) and labels (emotions).

### 2. Model Architecture
A CNN model was created using the Keras framework with the following layers:
Convolutional layers with ReLU activation to capture spatial features.
Max-pooling layers to downsample and reduce the spatial dimensions.
A fully connected layer with ReLU activation for higher-level feature learning.
The final output layer uses a softmax activation function for multi-class classification.
The model is compiled using the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.

### 3. Training
The dataset was split into training and testing sets.
Data augmentation techniques were used to increase the variety of the training data and prevent overfitting (rotation, shifting, zooming, etc.).
The model was trained for 15 epochs initially and further extended for additional epochs (20 epochs in total) with an adjusted learning rate.

## Results and Evaluation
Training Accuracy: The model's training accuracy improved over the epochs, demonstrating its learning process.
Validation Accuracy: The validation accuracy showed how well the model generalized to unseen data.
Test Accuracy: After training, the model was evaluated on the test set, achieving a test accuracy of approximately 70%.
The following insights were drawn from the training:

The model effectively learned to recognize facial expressions.
Adjusting the learning rate helped improve the model’s performance during extended training.

## Future Considerations
The current model achieved a 70% accuracy on the test set. However, there is room for improvement.

## Future work could include:
Exploring additional or more advanced CNN architectures.
Fine-tuning hyperparameters to optimize performance.
Experimenting with more advanced data augmentation techniques to further increase generalization.
Using other datasets or incorporating other techniques like Transfer Learning.

## Conclusion
This project demonstrates the potential of using Convolutional Neural Networks (CNNs) for facial emotion recognition. With further improvements and optimizations, the model could be deployed in applications requiring real-time emotion detection from facial expressions.
