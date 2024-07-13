import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bar

def load_images_from_folder(folder):
    images = []
    labels = []
    for class_folder in os.listdir(folder):
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):  # Ensure it's a directory
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (128, 128))  # Resize to a fixed size
                    images.append(img)
                    labels.append(class_folder)
    return np.array(images), np.array(labels)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0  # Normalize image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    return None

def predict_and_save_batch(image_paths, model, label_encoder, output_dir):
    for image_path in tqdm(image_paths, desc="Processing Images"):
        new_image = preprocess_image(image_path)
        if new_image is not None:
            predictions = model.predict(new_image)
            predicted_class = np.argmax(predictions, axis=1)
            predicted_label = label_encoder.inverse_transform(predicted_class)[0]

            # Create directory if it doesn't exist
            save_dir = os.path.join(output_dir, predicted_label)
            os.makedirs(save_dir, exist_ok=True)

            # Save the image in the predicted class directory
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, cv2.imread(image_path))
            print(f"PREDICTES AS: {predicted_label}")
        else:
            print(f"Failed to load image: {image_path}")

# Load images and labels
images, labels = load_images_from_folder(r'C:\Users\jainam dosi\Desktop\project\images')

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


image_folder = r'C:\Users\jainam dosi\Desktop\project\testimg1'  
output_dir = r'C:\Users\jainam dosi\Desktop\project\classified_images'  

image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg') or img.endswith('.png')]

predict_and_save_batch(image_paths, model, label_encoder, output_dir)
