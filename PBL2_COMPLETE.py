import os  # For operating system commands
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting results
from skimage.io import imread  # To read images from the file system
from keras import datasets, layers, models  # Keras for building the neural network

print('\n    Image Classification starting...')

print('### 1. Finding the directory ###')
print('Current working directory: ' + os.getcwd())  # Display the current working directory

# Define the input directory and subfolders for training and testing data
input_dir = '/content'
folders = ['permitted_Train', 'not permitted_Train', 'permitted_Test', 'not permitted_Test']

# Verify if all required folders exist
for folder in folders:
    if os.path.exists(os.path.join(input_dir, folder)):
        pass  # Proceed if the folder exists
    else:
        print(f"Error: No '{folder}' class in '{input_dir}'\n")
        quit()  # Exit the program if any folder is missing

print('Both directories exist. Proceed ...\n')

# Define training and testing categories
categories_train = ['permitted_Train', 'not permitted_Train']
categories_test = ['permitted_Test', 'not permitted_Test']

print("### 2. Image Resizing ###")
train_data, train_labels = [], []  # Lists to hold training data and labels
test_data, test_labels = [], []  # Lists to hold testing data and labels

# Process training data
for category_idx, category in enumerate(categories_train):
    category_path = os.path.join(input_dir, category)
    for file in os.listdir(category_path):
        img_path = os.path.join(category_path, file)  # Path to the image
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Check for valid image extensions
            try:
                img = imread(img_path)  # Read the image file
                img = np.resize(img, (32, 32, 3))  # Resize the image to 32x32 pixels with 3 color channels (RGB)
                train_data.append(img)  # Add the image to the training dataset
                train_labels.append(category_idx)  # Add the corresponding label
            except Exception as e:
                print(f"Error reading file {img_path}: {e}")  # Log error but continue
        else:
            print(f"Skipping non-image file: {file}")


# Process testing data
for category_idx, category in enumerate(categories_test):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)  # Path to the image
        img = imread(img_path)  # Read the image file
        img = np.resize(img, (32, 32, 3))  # Resize the image to 32x32 pixels with 3 color channels (RGB)
        test_data.append(img)  # Add the image to the testing dataset
        test_labels.append(category_idx)  # Add the corresponding label

# Convert datasets to NumPy arrays and normalize pixel values to the range [0, 1]
training_images = np.array(train_data).astype('float32') / 255.0
training_labels = np.array(train_labels)
testing_images = np.array(test_data).astype('float32') / 255.0
testing_labels = np.array(test_labels)

print('Finished resizing the images. Proceed ...\n')

# TODO choose the model (simple/moderate/complex) by uncomment the code
# Initialize a Sequential simple model
model = models.Sequential()  # Initialize a Sequential model

# Add layers to the model
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # Input Convolutional layer

model.add(layers.Conv2D(10, (3, 3), activation='relu'))  # Hidden convolutional layer
model.add(layers.Flatten())  # Flatten hidden layer to convert 3D output to 1D
model.add(layers.Dense(64, activation='relu'))  # Fully connected dense hidden layer

model.add(layers.Dense(2, activation='softmax'))  # Output layer with softmax activation (2 classes)

# Initialize a Sequential moderate model
# model = models.Sequential()  # Initialize a Sequential model

# Add layers to the model
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # Input Convolutional layer
#
# model.add(layers.Conv2D(50, (3, 3), activation='relu'))  # Hidden convolutional layer
# model.add(layers.Conv2D(50, (3, 3), activation='relu'))  # Hidden layer - pooling layer
# model.add(layers.Flatten())  # Flatten hidden layer to convert 3D output to 1D
# model.add(layers.Dense(64, activation='relu'))  # Fully connected dense hidden layer
#
# model.add(layers.Dense(2, activation='softmax'))  # Output layer with softmax activation (2 classes)

# Initialize a Sequential complex model
# model = models.Sequential()  # Initialize a Sequential model

# Add layers to the model
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # Input Convolutional layer
#
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))  # Hidden convolutional layer
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Hidden convolutional layer
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # Hidden layer - pooling layer
# model.add(layers.Flatten())  # Flatten hidden layer to convert 3D output to 1D
# model.add(layers.Dense(64, activation='relu'))  # Fully connected dense hidden layer
#
# model.add(layers.Dense(2, activation='softmax'))  # Output layer with softmax activation (2 classes)

model.compile(optimizer='adam',  # Use Adam optimizer
              loss='sparse_categorical_crossentropy',  # Loss function for classification tasks
              metrics=['accuracy'])  # Metric to monitor during training

history = model.fit(training_images, training_labels, epochs=10,  # Train for 10 epochs
                    validation_data=(testing_images, testing_labels))  # Use testing data for validation

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Evaluate the model on the testing dataset
loss, accuracy = model.evaluate(testing_images, testing_labels)  # Evaluate on testing data
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Get predictions for the test dataset
predicted_probs = model.predict(testing_images)
predicted_classes = np.argmax(predicted_probs, axis=1)  # Convert probabilities to class labels

# Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
class_names = ['Permitted', 'Not Permitted']  # Adjust according to your classes
report = classification_report(testing_labels, predicted_classes, target_names=class_names)
print(report)

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()