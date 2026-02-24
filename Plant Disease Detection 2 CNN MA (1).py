# src/cnn_model.py
"""
CNN Model Architecture for Plant Disease Detection
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense,
    Dropout, BatchNormalization, Input, Concatenate,
    Activation, DepthwiseConv2D
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, Dict, List, Optional
import numpy as np
import os

class PlantDiseaseCNN:
    """
    CNN Model for Plant Disease Detection
    Supports multiple architectures: Custom CNN, MobileNetV2, EfficientNet
    """
    
    # Disease classes for Plant Village dataset
    DISEASE_CLASSES = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry___healthy',
        'Cherry___Powdery_mildew',
        'Corn___Cercospora_leaf_spot',
        'Corn___Common_rust',
        'Corn___healthy',
        'Corn___Northern_Leaf_Blight',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___healthy',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___healthy',
        'Potato___Late_blight',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___healthy',
        'Strawberry___Leaf_scorch',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___healthy',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites',
        'Tomato___Target_Spot',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
    ]
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = None, model_type: str = 'custom'):
        """
        Initialize the CNN model
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of disease classes
            model_type: Type of model architecture ('custom', 'mobilenet', 'efficientnet')
        """
        self.input_shape = input_shape
        self.num_classes = num_classes or len(self.DISEASE_CLASSES)
        self.model_type = model_type
        self.model = None
        self.history = None
        
        # Build model based on type
        if model_type == 'custom':
            self.model = self._build_custom_cnn()
        elif model_type == 'mobilenet':
            self.model = self._build_mobilenet_model()
        elif model_type == 'efficientnet':
            self.model = self._build_efficientnet_model()
        else:
            self.model = self._build_custom_cnn()
    
    def _build_custom_cnn(self) -> Sequential:
        """Build custom CNN architecture"""
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                   input_shape=self.input_shape),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(0.25),
            
            # Block 4
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(0.25),
            
            # Classifier
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_mobilenet_model(self) -> Model:
        """Build MobileNetV2 based model with transfer learning"""
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        # Add custom classifier
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model
    
    def _build_efficientnet_model(self) -> Model:
        """Build EfficientNetB0 based model with transfer learning"""
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        # Add custom classifier
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model
    
    def compile(self, learning_rate: float = 0.001):
        """Compile the model"""
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        print(f"Model compiled with learning rate: {learning_rate}")
    
    def get_callbacks(self, checkpoint_path: str, log_dir: str) -> List:
        """Get training callbacks"""
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_path, 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True
            ),
            
            # CSV Logger
            CSVLogger(os.path.join(log_dir, 'training_log.csv'))
        ]
        
        return callbacks
    
    def train(self, train_generator, val_generator,
              epochs: int = 50, checkpoint_path: str = 'models/checkpoints',
              log_dir: str = 'logs') -> Dict:
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of training epochs
            checkpoint_path: Path to save checkpoints
            log_dir: Path for training logs
            
        Returns:
            Training history
        """
        callbacks = self.get_callbacks(checkpoint_path, log_dir)
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_steps=val_generator.samples // val_generator.batch_size
        )
        
        return self.history.history
    
    def evaluate(self, test_generator) -> Dict:
        """Evaluate model on test data"""
        evaluation = self.model.evaluate(
            test_generator,
            steps=test_generator.samples // test_generator.batch_size
        )
        
        return dict(zip(self.model.metrics_names, evaluation))
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Predict disease for a single image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Dictionary with predictions
        """
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[::-1][:5]
        top_predictions = [
            {
                'class': self.DISEASE_CLASSES[idx],
                'confidence': float(predictions[0][idx]),
                'plant': self.DISEASE_CLASSES[idx].split('___')[0],
                'disease': self.DISEASE_CLASSES[idx].split('___')[1].replace('_', ' ')
            }
            for idx in top_indices
        ]
        
        return {
            'top_predictions': top_predictions,
            'predicted_class': top_predictions[0]['class'],
            'confidence': top_predictions[0]['confidence']
        }
    
    def get_feature_maps(self, image: np.ndarray, layer_name: str = None):
        """Get feature maps for visualization"""
        # Create a model that outputs intermediate layers
        if layer_name is None:
            layer_name = 'conv2d_3'  # Default to an early conv layer
        
        layer = self.model.get_layer(layer_name)
        activation_model = Model(
            inputs=self.model.input,
            outputs=layer.output
        )
        
        # Get activations
        activations = activation_model.predict(np.expand_dims(image, axis=0), verbose=0)
        
        return activations
    
    def save_model(self, filepath: str):
        """Save model to file"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary"""
        self.model.summary()
    
    def unfreeze_layers(self, num_layers: int = 20):
        """Unfreeze last N layers for fine-tuning"""
        for layer in self.model.layers[-num_layers:]:
            layer.trainable = True
        
        print(f"Unfroze last {num_layers} layers for fine-tuning")


class ModelEvaluator:
    """Evaluate and analyze model performance"""
    
    def __init__(self, model: PlantDiseaseCNN, class_names: List[str]):
        self.model = model
        self.class_names = class_names
    
    def plot_training_history(self, history: Dict, save_path: str = None):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Accuracy
        axes[0].plot(history['accuracy'], label='Train')
        axes[0].plot(history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        # Loss
        axes[1].plot(history['loss'], label='Train')
        axes[1].plot(history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        # Precision & Recall
        axes[2].plot(history['precision'], label='Precision')
        axes[2].plot(history['recall'], label='Recall')
        axes[2].set_title('Precision & Recall')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Score')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              save_path: str = None):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return plt.gcf()
    
    def plot_class_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                               save_path: str = None):
        """Plot per-class performance"""
        from sklearn.metrics import classification_report
        import matplotlib.pyplot as plt
        import pandas as pd
        
        report = classification_report(y_true, y_pred, 
                                      target_names=self.class_names,
                                      output_dict=True)
        
        # Extract metrics for each class
        metrics_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': [report[c]['precision'] for c in self.class_names],
            'Recall': [report[c]['recall'] for c in self.class_names],
            'F1-Score': [report[c]['f1-score'] for c in self.class_names]
        })
        
        fig, ax = plt.subplots(figsize=(16, 10))
        metrics_df.plot(x='Class', kind='bar', ax=ax)
        plt.title('Per-Class Performance')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.xticks(rotation=90, fontsize=8)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig