"""
Test Intent Extraction and Onboarding Document Generation System - Root Directory Initialization
Five-layer architecture: Input → Extract → Intent → Business → Output
"""

__version__ = "1.0.0"
__author__ = "Test Intent Extraction System"
__description__ = "Automated test intent extraction and onboarding document generation"

# Import all layers
from layers import InputLayer, ExtractLayer, IntentLayer, BusinessLayer, OutputLayer

__all__ = [
    'InputLayer',
    'ExtractLayer', 
    'IntentLayer',
    'BusinessLayer',
    'OutputLayer',
]
