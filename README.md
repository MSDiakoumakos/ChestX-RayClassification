# ChestX-Ray Classification
This is a machine learning project focused on medical image classification using deep learning techniques. The project aims to classify chest X-ray images into 4 different categories:
 1 - Normal - Healthy chest X-rays
 2 - COVID - COVID-19 positive cases
 3 - Viral Pneumonia - Other viral pneumonia cases
 4 - Lung_Opacity - Cases with lung opacity
## Key Features:
Dataset: Uses the COVID-19 Radiography Dataset with chest X-ray images
## Models Implemented:
CNN1: Simple convolutional neural network with 2 conv layers
CNN2: More complex CNN with 5 blocks and padding
ResNet50: Pre-trained ResNet50 with transfer learning
CustomResNet: Custom residual network implementation
## Technical Approach:
Built with PyTorch framework
Implements custom dataset class for medical images
Includes data visualization and class distribution analysis
Features early stopping to prevent overfitting
Uses Adam optimizer with different learning rates
Implements confusion matrix for performance evaluation
