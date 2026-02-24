# src/disease_detector.py
"""
Disease Detection Engine
Main interface for plant disease detection
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .cnn_model import PlantDiseaseCNN
from .data_generator import ImagePreprocessor

class DiseaseDetector:
    """
    Main disease detection interface
    """
    
    def __init__(self, model_path: str = None, model_type: str = 'custom'):
        """
        Initialize disease detector
        
        Args:
            model_path: Path to trained model
            model_type: Type of model architecture
        """
        self.model = None
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            # Initialize with default architecture
            self.model = PlantDiseaseCNN(model_type=model_type)
            self.model.compile()
    
    def load_model(self, model_path: str):
        """Load trained model"""
        self.model = PlantDiseaseCNN()
        self.model.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def detect(self, image: np.ndarray, top_k: int = 5) -> Dict:
        """
        Detect disease in an image
        
        Args:
            image: Input image (RGB format)
            top_k: Number of top predictions to return
            
        Returns:
            Detection results dictionary
        """
        # Preprocess image
        processed_image = self.preprocessor.normalize(image)
        processed_image = cv2.resize(processed_image, (224, 224))
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        
        # Format results
        results = {
            'predictions': predictions['top_predictions'][:top_k],
            'primary_prediction': predictions['predicted_class'],
            'confidence': predictions['confidence'],
            'plant': predictions['top_predictions'][0]['plant'],
            'disease': predictions['top_predictions'][0]['disease']
        }
        
        return results
    
    def detect_from_path(self, image_path: str, top_k: int = 5) -> Dict:
        """
        Detect disease from image file path
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions
            
        Returns:
            Detection results
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self.detect(image, top_k)
    
    def detect_from_bytes(self, image_bytes: bytes, top_k: int = 5) -> Dict:
        """
        Detect disease from image bytes
        
        Args:
            image_bytes: Image data as bytes
            top_k: Number of top predictions
            
        Returns:
            Detection results
        """
        # Load image from bytes
        image = Image.open(image_bytes)
        image = np.array(image)
        
        return self.detect(image, top_k)
    
    def get_treatment_recommendation(self, disease: str) -> Dict:
        """
        Get treatment recommendation for detected disease
        
        Args:
            disease: Disease name
            
        Returns:
            Treatment recommendations
        """
        recommendations = {
            'Apple___Apple_scab': {
                'treatment': 'Apply fungicides containing captan or mancozeb',
                'prevention': 'Plant resistant varieties, remove infected leaves',
                'organic': 'Neem oil, copper-based fungicides'
            },
            'Apple___Black_rot': {
                'treatment': 'Prune infected branches, apply fungicides',
                'prevention': 'Remove mummified fruit, proper sanitation',
                'organic': 'Copper spray, biological controls'
            },
            'Apple___Cedar_apple_rust': {
                'treatment': 'Remove cedar trees nearby, apply fungicides',
                'prevention': 'Plant resistant varieties',
                'organic': 'Sulfur-based sprays'
            },
            'Corn___Common_rust': {
                'treatment': 'Apply fungicides if severe',
                'prevention': 'Plant resistant hybrids, crop rotation',
                'organic': 'Neem oil, biological fungicides'
            },
            'Corn___Northern_Leaf_Blight': {
                'treatment': 'Fungicide application at disease onset',
                'prevention': 'Resistant hybrids, tillage to reduce residue',
                'organic': 'Crop rotation, biological controls'
            },
            'Potato___Late_blight': {
                'treatment': 'Immediate fungicide application',
                'prevention': 'Remove volunteer plants, proper spacing',
                'organic': 'Copper-based fungicides, proper irrigation'
            },
            'Tomato___Late_blight': {
                'treatment': 'Remove infected plants, apply fungicides',
                'prevention': 'Good air circulation, avoid overhead watering',
                'organic': 'Copper sprays, compost tea'
            },
            'Tomato___Leaf_Mold': {
                'treatment': 'Improve ventilation, apply fungicides',
                'prevention': 'Reduce humidity, proper spacing',
                'organic': 'Neem oil, sulfur-based sprays'
            }
        }
        
        # Default recommendation
        default = {
            'treatment': 'Consult local agricultural extension',
            'prevention': 'Remove infected plant parts, improve air circulation',
            'organic': 'Neem oil, copper-based products'
        }
        
        return recommendations.get(disease, default)
    
    def visualize_detection(self, image: np.ndarray, results: Dict,
                           save_path: str = None) -> np.ndarray:
        """
        Visualize detection results on image
        
        Args:
            image: Original image
            results: Detection results
            save_path: Path to save visualization
            
        Returns:
            Visualization image
        """
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display image
        ax.imshow(image)
        
        # Add title with prediction
        title = f"Primary: {results['primary_prediction']}\n"
        title += f"Confidence: {results['confidence']:.2%}"
        ax.set_title(title, fontsize=14)
        
        # Add prediction bar
        predictions = results['predictions'][:5]
        y_pos = np.arange(len(predictions))
        confidences = [p['confidence'] for p in predictions]
        labels = [p['disease'] for p in predictions]
        
        ax.barh(y_pos, confidences, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('Confidence')
        ax.set_title('Top 5 Predictions')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def batch_detect(self, image_paths: List[str], top_k: int = 5) -> List[Dict]:
        """
        Detect diseases in multiple images
        
        Args:
            image_paths: List of image paths
            top_k: Number of top predictions
            
        Returns:
            List of detection results
        """
        results = []
        
        for path in image_paths:
            try:
                result = self.detect_from_path(path, top_k)
                result['image_path'] = path
                results.append(result)
            except Exception as e:
                results.append({
                    '