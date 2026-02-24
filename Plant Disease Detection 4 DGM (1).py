# src/data_generator.py
"""
Data Augmentation and Generation Module
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from typing import Tuple, List, Dict, Optional, Generator
from pathlib import Path
import cv2
import os
import random

class DataGenerator:
    """
    Data augmentation and generation for plant disease detection
    """
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize data generator
        
        Args:
            img_size: Target image size (width, height)
        """
        self.img_size = img_size
    
    def get_train_datagen(self, augmentation: bool = True) -> ImageDataGenerator:
        """
        Get training data generator with augmentation
        
        Args:
            augmentation: Whether to apply data augmentation
            
        Returns:
            ImageDataGenerator instance
        """
        if augmentation:
            return ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest',
                brightness_range=[0.8, 1.2],
                channel_shift_range=0.1
            )
        else:
            return ImageDataGenerator(rescale=1./255)
    
    def get_val_datagen(self) -> ImageDataGenerator:
        """Get validation data generator"""
        return ImageDataGenerator(rescale=1./255)
    
    def get_test_datagen(self) -> ImageDataGenerator:
        """Get test data generator"""
        return ImageDataGenerator(rescale=1./255)
    
    def flow_from_directory(self, directory: str, datagen: ImageDataGenerator,
                           batch_size: int = 32, target_size: Tuple[int, int] = None,
                           class_mode: str = 'categorical', shuffle: bool = True):
        """
        Create data generator from directory
        
        Args:
            directory: Path to image directory
            datagen: ImageDataGenerator instance
            batch_size: Batch size
            target_size: Target image size
            class_mode: Classification mode
            shuffle: Whether to shuffle data
            
        Returns:
            Data generator
        """
        target_size = target_size or self.img_size
        
        return datagen.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=shuffle
        )
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Load image
        img = load_img(image_path, target_size=self.img_size)
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Rescale
        img_array = img_array / 255.0
        
        return img_array
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to a single image
        
        Args:
            image: Input image array
            
        Returns:
            Augmented image array
        """
        datagen = self.get_train_datagen()
        
        # Expand dimensions
        img_array = np.expand_dims(image, axis=0)
        
        # Generate augmented image
        aug_iter = datagen.flow(img_array, batch_size=1)
        
        return next(aug_iter)[0]
    
    def create_balanced_dataset(self, images: List[str], labels: List[int],
                                target_count: int = None) -> Tuple[List[str], List[int]]:
        """
        Create balanced dataset using oversampling
        
        Args:
            images: List of image paths
            labels: List of labels
            target_count: Target samples per class
            
        Returns:
            Balanced image paths and labels
        """
        from collections import Counter
        
        # Count samples per class
        class_counts = Counter(labels)
        
        # Determine target count
        if target_count is None:
            target_count = max(class_counts.values())
        
        balanced_images = []
        balanced_labels = []
        
        for class_label, count in class_counts.items():
            # Get images for this class
            class_images = [img for img, lbl in zip(images, labels) if lbl == class_label]
            
            # Oversample if needed
            while len(class_images) < target_count:
                class_images.extend(class_images[:min(target_count - len(class_images), len(class_images))])
            
            # Select samples
            selected = class_images[:target_count]
            balanced_images.extend(selected)
            balanced_labels.extend([class_label] * len(selected))
        
        return balanced_images, balanced_labels
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract features from image for analysis
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Convert to different color spaces
        rgb = image
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Basic statistics
        features['mean_intensity'] = np.mean(rgb)
        features['std_intensity'] = np.std(rgb)
        
        # Color channel statistics
        for i, channel in enumerate(['red', 'green', 'blue']):
            features[f'{channel}_mean'] = np.mean(rgb[:, :, i])
            features[f'{channel}_std'] = np.std(rgb[:, :, i])
        
        # HSV statistics
        for i, channel in enumerate(['hue', 'saturation', 'value']):
            features[f'{channel}_mean'] = np.mean(hsv[:, :, i])
            features[f'{channel}_std'] = np.std(hsv[:, :, i])
        
        # Texture features (simple variance-based)
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        features['texture_variance'] = np.var(gray)
        features['texture_energy'] = np.sum(gray ** 2) / (gray.shape[0] * gray.shape[1])
        
        return features


class ImagePreprocessor:
    """Image preprocessing utilities"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size
        """
        self.target_size = target_size
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(image, self.target_size)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        return image / 255.0
    
    def standardize(self, image: np.ndarray, mean: List = None, std: List = None) -> np.ndarray:
        """
        Standardize image using ImageNet statistics
        
        Args:
            image: Input image
            mean: Mean values for each channel
            std: Standard deviation values for each channel
        """
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]
        
        image = image.astype(np.float32)
        
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
        return image
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Useful for improving leaf vein visibility
        """
        # Convert to LAB
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result.astype(np.float32) / 255.0
    
    def remove_background(self, image: np.ndarray, threshold: int = 200) -> np.ndarray:
        """
        Remove background using color threshold
        
        Args:
            image: Input image
            threshold: Threshold for background removal
            
        Returns:
            Image with background removed
        """
        mask = np.all(image > (threshold / 255), axis=-1)
        result = image.copy()
        result[mask] = [1, 1, 1]  # Set background to white
        
        return result
    
    def segment_leaf(self, image: np.ndarray) -> np.ndarray:
        """
        Simple leaf segmentation using color thresholding
        
        Args:
            image: Input image
            
        Returns:
            Segmented image
        """
        # Convert to HSV
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Define green color range
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return result
    
    def enhance_for_disease(self, image: np.ndarray) -> np.ndarray:
        """
        Apply disease-specific enhancements
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Apply CLAHE
        enhanced = self.apply_clahe(image)
        
        # Slight saturation boost for better disease spot visibility
        hsv = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return enhanced.astype(np.float32) / 255.0