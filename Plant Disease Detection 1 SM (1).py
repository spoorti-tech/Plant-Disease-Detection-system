# src/__init__.py
"""
Plant Disease Detection System
Deep Learning based plant disease identification
"""

__version__ = "1.0.0"
__author__ = "Deep Learning Engineer"

from .cnn_model import PlantDiseaseCNN
from .disease_detector import DiseaseDetector
from .data_generator import DataGenerator
from .plant_village import PlantVillageDataset