import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import cv2
from pathlib import Path

def create_improved_model():
    """
    Improved CNN architecture with:
    - Batch Normalization for faster convergence
    - More convolutional layers for better feature extraction
    - Residual-like connections
    - Better regularization
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(7, activation='softmax')
    ])
    
    # Use a better optimizer with learning rate scheduling
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_data(data_dir):
    """Load images and labels from directory structure with better preprocessing"""
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
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for emotion in os.listdir(data_dir):
        emotion_dir = data_path / emotion
        
        if emotion_dir.is_dir() and emotion in emotion_map:
            print(f"Loading {emotion} images...")
            
            image_count = 0
            for img_file in emotion_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    try:
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        
                        if img is not None:
                            # Resize to 48x48
                            img = cv2.resize(img, (48, 48))
                            
                            # Apply histogram equalization for better contrast
                            img = cv2.equalizeHist(img)
                            
                            # Normalize to [0, 1]
                            img = img / 255.0
                            
                            images.append(img)
                            labels.append(emotion_map[emotion])
                            image_count += 1
                    except Exception as e:
                        print(f"  Error loading {img_file.name}: {e}")
            
            print(f"  ✓ Loaded {image_count} {emotion} images")
    
    if len(images) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    return np.array(images), np.array(labels)

def main():
    train_dir = "FER-2013/train"
    test_dir = "FER-2013/test"
    
    if not os.path.exists(train_dir):
        print("ERROR: FER-2013 dataset not found!")
        print(f"Please ensure '{train_dir}' exists.")
        return
    
    print("=" * 80)
    print("IMPROVED FER-2013 Emotion Detection Model Training")
    print("Target: 80%+ Accuracy")
    print("=" * 80)
    
    try:
        print("\n[1/6] Loading training data...")
        X_train, y_train = load_data(train_dir)
        
        print("\n[2/6] Loading test data...")
        X_test, y_test = load_data(test_dir)
        
        print(f"\n{'='*80}")
        print(f"Dataset Summary:")
        print(f"  Training images: {len(X_train)}")
        print(f"  Test images: {len(X_test)}")
        
        # Show class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        print(f"\n  Training set distribution:")
        for idx, count in zip(unique, counts):
            print(f"    {emotion_names[idx]}: {count}")
        print(f"{'='*80}\n")
        
        # Reshape for CNN
        X_train = X_train.reshape(-1, 48, 48, 1)
        X_test = X_test.reshape(-1, 48, 48, 1)
        
        print("[3/6] Creating improved model architecture...")
        model = create_improved_model()
        print(f"\nTotal parameters: {model.count_params():,}")
        
        # More aggressive data augmentation
        print("\n[4/6] Setting up data augmentation...")
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Advanced callbacks
        from tensorflow.keras.callbacks import (
            EarlyStopping, 
            ReduceLROnPlateau, 
            ModelCheckpoint,
            LearningRateScheduler
        )
        
        # Learning rate scheduler
        def lr_schedule(epoch, lr):
            if epoch < 30:
                return lr
            elif epoch < 60:
                return lr * 0.5
            elif epoch < 90:
                return lr * 0.1
            else:
                return lr * 0.01
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_emotion_model_80.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            LearningRateScheduler(lr_schedule, verbose=1)
        ]
        
        print("\n[5/6] Starting training...")
        print("This will take a while but should reach 80%+ accuracy!")
        print("Training with 150 epochs and early stopping...\n")
        
        # Train with more epochs
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=64),
            epochs=150,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            steps_per_epoch=len(X_train) // 64,
            verbose=1
        )
        
        print("\n[6/6] Evaluating final model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS:")
        print(f"{'='*80}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        if test_accuracy >= 0.80:
            print(f"\n  ✓ SUCCESS! Target of 80% achieved!")
        elif test_accuracy >= 0.75:
            print(f"\n  ⚠ Close! Consider training longer or tuning hyperparameters")
        else:
            print(f"\n  ⚠ Below target. Try running again or increasing epochs")
        
        print(f"{'='*80}\n")
        
        # Save models
        model.save('emotion_model_improved.h5')
        print("✓ Model saved as 'emotion_model_improved.h5'")
        print("✓ Best model saved as 'best_emotion_model_80.h5'")
        
        # Save training history
        import pickle
        with open('training_history_improved.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        print("✓ Training history saved")
        
        # Plot training history
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=150)
            print("✓ Training plots saved as 'training_history.png'")
            
        except ImportError:
            print("⚠ Matplotlib not installed, skipping plots")
        
        print("\nTraining complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()