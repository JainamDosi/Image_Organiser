import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bar
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_from_csv(csv_file, base_path):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = os.path.join(base_path, row['Image'])  
        label = row['Class']
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))  
            images.append(img)
            labels.append(label)
        else:
            print(f"Failed to load image: {img_path}")
    return np.array(images), np.array(labels)

def preprocess_image(image):
    img = cv2.resize(image, (128, 128))
    img = img.astype('float32') / 255.0  # Normalize image
    return img

def predict_and_save_batch(image_paths, model, label_encoder, output_dir):
    for image_path in tqdm(image_paths, desc="Processing Images"):
        img = cv2.imread(image_path)
        if img is not None:
            new_image = preprocess_image(img)
            new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension
            predictions = model.predict(new_image)
            predicted_class = np.argmax(predictions, axis=1)
            predicted_label = label_encoder.inverse_transform(predicted_class)[0]

            # Create directory if it doesn't exist
            save_dir = os.path.join(output_dir, predicted_label)
            os.makedirs(save_dir, exist_ok=True)

            # Save the image in the predicted class directory
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, img)
        else:
            print(f"Failed to load image: {image_path}")

# Load images and labels from CSV
# Example usage of load_images_from_csv function
csv_file = r'C:\Users\jainam dosi\Desktop\project\archive\dataset\train.csv'
base_path = r'C:\Users\jainam dosi\Desktop\project\archive\dataset\train\\'

images, labels = load_images_from_csv(csv_file, base_path)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Normalize images
images = images.astype('float32') / 255.0

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
val_generator = val_datagen.flow(val_images, val_labels, batch_size=32)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(labels_encoded)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
model.fit(train_generator, epochs=50, validation_data=val_generator)

# Save the model
model.save(r'C:\Users\jainam dosi\Desktop\project\model.h5')

# Load the model
model = load_model(r'C:\Users\jainam dosi\Desktop\project\model.h5')

# Example usage to classify and save a batch of images
image_folder = r'C:\Users\jainam dosi\Desktop\project\archive\dataset\test'  # Replace with the folder containing images
output_dir = r'C:\Users\jainam dosi\Desktop\project\classified_images'   # Replace with your desired output directory

image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg') or img.endswith('.png')]

predict_and_save_batch(image_paths, model, label_encoder, output_dir)
