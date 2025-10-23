import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def load_data(data_dir):
    images = []
    labels = []
    emotion_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    
    # Walk through the data directory
    for emotion in os.listdir(data_dir):
        emotion_dir = os.path.join(data_dir, emotion)
        if os.path.isdir(emotion_dir):
            print(f"Loading {emotion} images...")
            for img_name in os.listdir(emotion_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(emotion_dir, img_name)
                    # Read and preprocess image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (48, 48))
                        img = img / 255.0  # Normalize
                        images.append(img)
                        labels.append(emotion_map[emotion])
    
    return np.array(images), np.array(labels)

def main():
    print("Loading training data...")
    train_dir = "FER-2013/train"
    X_train, y_train = load_data(train_dir)
    
    print("Loading test data...")
    test_dir = "FER-2013/test"
    X_test, y_test = load_data(test_dir)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("No images found! Please check your data directory structure.")
        return
    
    print(f"Loaded {len(X_train)} training images and {len(X_test)} test images")
    
    # Reshape for CNN
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)
    
    print("Creating and training model...")
    model = create_model()
    
    # Add data augmentation
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Add callbacks for better training
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    # Train the model with increased epochs and data augmentation
    history = model.fit(datagen.flow(X_train, y_train, batch_size=128),
                       epochs=100,
                       validation_data=(X_test, y_test),
                       callbacks=[early_stopping, reduce_lr],
                     steps_per_epoch=len(X_train) // 128)
    
    # Evaluate the model on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save the model
    model.save('emotion_model.h5')
    print("Model saved as 'emotion_model.h5'")

if __name__ == '__main__':
    main() 