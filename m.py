#!/usr/bin/env python3
"""
Main Model Module (m.py)
Core processing and model management for the AI Image Generator
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st
import os
import tempfile
import base64
import io
from datetime import datetime
import time

# Import project modules
from image_processor import ImageProcessor
from filters import BasicFilters, CreativeFilters, AIEffects, FilterProcessor
from utils import (
    create_temp_file, save_uploaded_file, image_to_bytes, 
    get_image_download_link, validate_image, optimize_image_size
)
from config import *

class AIModel:
    """
    Main AI Model class that orchestrates all image processing operations
    """
    
    def __init__(self):
        """Initialize the AI Model with all components"""
        self.image_processor = ImageProcessor()
        self.filter_processor = FilterProcessor()
        self.basic_filters = BasicFilters()
        self.creative_filters = CreativeFilters()
        self.ai_effects = AIEffects()
        
        # Processing state
        self.current_image = None
        self.processed_image = None
        self.processing_history = []
        
        # Performance tracking
        self.processing_times = {}
        
    def load_and_validate_image(self, uploaded_file):
        """
        Load and validate uploaded image
        
        Args:
            uploaded_file: StreamlitUploadedFile - Uploaded image file
            
        Returns:
            tuple: (success: bool, image_array: numpy.ndarray, message: str)
        """
        try:
            # Check file size
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                return False, None, f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB"
            
            # Check file type
            if uploaded_file.type not in ALLOWED_FILE_TYPES:
                return False, None, f"Unsupported file type: {uploaded_file.type}"
            
            # Load image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Validate image
            if not validate_image(image_array):
                return False, None, "Invalid image format"
            
            # Optimize size
            image_array = optimize_image_size(image_array, MAX_IMAGE_SIZE)
            
            self.current_image = image_array
            return True, image_array, "Image loaded successfully"
            
        except Exception as e:
            return False, None, f"Error loading image: {str(e)}"
    
    def apply_basic_filter(self, filter_name, **parameters):
        """
        Apply basic filter to current image
        
        Args:
            filter_name: str - Name of filter to apply
            **parameters: Filter-specific parameters
            
        Returns:
            tuple: (success: bool, result_image: numpy.ndarray, message: str)
        """
        if self.current_image is None:
            return False, None, "No image loaded"
        
        try:
            start_time = time.time()
            
            if filter_name == "grayscale":
                result = self.basic_filters.grayscale(self.current_image)
            elif filter_name == "blur":
                kernel_size = parameters.get('kernel_size', DEFAULT_BLUR_KERNEL)
                result = self.basic_filters.blur(self.current_image, kernel_size)
            elif filter_name == "edge_detection":
                threshold1 = parameters.get('threshold1', DEFAULT_EDGE_THRESHOLD1)
                threshold2 = parameters.get('threshold2', DEFAULT_EDGE_THRESHOLD2)
                result = self.basic_filters.edge_detection(self.current_image, threshold1, threshold2)
            elif filter_name == "sepia":
                intensity = parameters.get('intensity', DEFAULT_SEPIA_INTENSITY)
                result = self.basic_filters.sepia(self.current_image, intensity)
            elif filter_name == "invert":
                result = self.basic_filters.invert(self.current_image)
            else:
                return False, None, f"Unknown basic filter: {filter_name}"
            
            processing_time = time.time() - start_time
            self.processing_times[filter_name] = processing_time
            
            self.processed_image = result
            self._add_to_history(filter_name, parameters)
            
            return True, result, f"{filter_name} applied successfully"
            
        except Exception as e:
            return False, None, f"Error applying {filter_name}: {str(e)}"
    
    def apply_creative_filter(self, filter_name, **parameters):
        """
        Apply creative filter to current image
        
        Args:
            filter_name: str - Name of filter to apply
            **parameters: Filter-specific parameters
            
        Returns:
            tuple: (success: bool, result_image: numpy.ndarray, message: str)
        """
        if self.current_image is None:
            return False, None, "No image loaded"
        
        try:
            start_time = time.time()
            
            if filter_name == "cartoonify":
                num_bilateral = parameters.get('num_bilateral', DEFAULT_CARTOON_BILATERAL)
                num_dilation = parameters.get('num_dilation', DEFAULT_CARTOON_DILATION)
                result = self.creative_filters.cartoonify(self.current_image, num_bilateral, num_dilation)
            elif filter_name == "pencil_sketch":
                blur_factor = parameters.get('blur_factor', DEFAULT_SKETCH_BLUR)
                result = self.creative_filters.pencil_sketch(self.current_image, blur_factor)
            elif filter_name == "oil_painting":
                radius = parameters.get('radius', DEFAULT_OIL_RADIUS)
                intensity = parameters.get('intensity', DEFAULT_OIL_INTENSITY)
                result = self.creative_filters.oil_painting(self.current_image, radius, intensity)
            elif filter_name == "watercolor":
                blur_radius = parameters.get('blur_radius', DEFAULT_WATERCOLOR_BLUR)
                edge_strength = parameters.get('edge_strength', DEFAULT_WATERCOLOR_EDGE)
                result = self.creative_filters.watercolor(self.current_image, blur_radius, edge_strength)
            else:
                return False, None, f"Unknown creative filter: {filter_name}"
            
            processing_time = time.time() - start_time
            self.processing_times[filter_name] = processing_time
            
            self.processed_image = result
            self._add_to_history(filter_name, parameters)
            
            return True, result, f"{filter_name} applied successfully"
            
        except Exception as e:
            return False, None, f"Error applying {filter_name}: {str(e)}"
    
    def apply_ai_effect(self, effect_name, **parameters):
        """
        Apply AI effect to current image
        
        Args:
            effect_name: str - Name of AI effect to apply
            **parameters: Effect-specific parameters
            
        Returns:
            tuple: (success: bool, result_image: numpy.ndarray, message: str)
        """
        if self.current_image is None:
            return False, None, "No image loaded"
        
        try:
            start_time = time.time()
            
            if effect_name == "neural_style_transfer":
                style_image_path = parameters.get('style_image_path')
                result = self.ai_effects.neural_style_transfer(self.current_image, style_image_path)
            elif effect_name == "artistic_style":
                style_type = parameters.get('style_type', 'vangogh')
                result = self.ai_effects.artistic_style(self.current_image, style_type)
            elif effect_name == "photo_to_art":
                art_style = parameters.get('art_style', 'impressionist')
                result = self.ai_effects.photo_to_art(self.current_image, art_style)
            else:
                return False, None, f"Unknown AI effect: {effect_name}"
            
            processing_time = time.time() - start_time
            self.processing_times[effect_name] = processing_time
            
            self.processed_image = result
            self._add_to_history(effect_name, parameters)
            
            return True, result, f"{effect_name} applied successfully"
            
        except Exception as e:
            return False, None, f"Error applying {effect_name}: {str(e)}"
    
    def apply_filter_chain(self, filter_chain):
        """
        Apply multiple filters in sequence
        
        Args:
            filter_chain: list - List of filter dictionaries with name and parameters
            
        Returns:
            tuple: (success: bool, result_image: numpy.ndarray, message: str)
        """
        if self.current_image is None:
            return False, None, "No image loaded"
        
        try:
            current_image = self.current_image.copy()
            
            for filter_config in filter_chain:
                filter_name = filter_config['name']
                parameters = filter_config.get('parameters', {})
                
                # Apply filter based on type
                if filter_name in ["grayscale", "blur", "edge_detection", "sepia", "invert"]:
                    success, result, message = self.apply_basic_filter(filter_name, **parameters)
                elif filter_name in ["cartoonify", "pencil_sketch", "oil_painting", "watercolor"]:
                    success, result, message = self.apply_creative_filter(filter_name, **parameters)
                elif filter_name in ["neural_style_transfer", "artistic_style", "photo_to_art"]:
                    success, result, message = self.apply_ai_effect(filter_name, **parameters)
                else:
                    return False, None, f"Unknown filter: {filter_name}"
                
                if not success:
                    return False, None, message
                
                current_image = result
            
            self.processed_image = current_image
            return True, current_image, "Filter chain applied successfully"
            
        except Exception as e:
            return False, None, f"Error applying filter chain: {str(e)}"
    
    def get_processing_stats(self):
        """
        Get processing statistics
        
        Returns:
            dict: Processing statistics
        """
        stats = {
            'total_operations': len(self.processing_history),
            'processing_times': self.processing_times,
            'average_time': np.mean(list(self.processing_times.values())) if self.processing_times else 0,
            'last_operation': self.processing_history[-1] if self.processing_history else None
        }
        return stats
    
    def save_processed_image(self, filename=None, format='PNG'):
        """
        Save processed image to file
        
        Args:
            filename: str - Output filename
            format: str - Output format
            
        Returns:
            tuple: (success: bool, file_path: str, message: str)
        """
        if self.processed_image is None:
            return False, None, "No processed image to save"
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ai_generated_{timestamp}.{format.lower()}"
            
            # Convert to PIL Image
            if len(self.processed_image.shape) == 3 and self.processed_image.shape[2] == 3:
                pil_image = Image.fromarray(self.processed_image, 'RGB')
            else:
                pil_image = Image.fromarray(self.processed_image, 'L')
            
            # Save to temporary file
            temp_path = create_temp_file(suffix=f'.{format.lower()}')
            pil_image.save(temp_path, format=format, quality=EXPORT_QUALITY)
            
            return True, temp_path, f"Image saved as {filename}"
            
        except Exception as e:
            return False, None, f"Error saving image: {str(e)}"
    
    def get_download_link(self, filename="ai_generated_image.png"):
        """
        Generate download link for processed image
        
        Args:
            filename: str - Download filename
            
        Returns:
            str: HTML download link
        """
        if self.processed_image is None:
            return None
        
        try:
            return get_image_download_link(self.processed_image, filename)
        except Exception as e:
            st.error(f"Error generating download link: {str(e)}")
            return None
    
    def reset_processing(self):
        """Reset processing state"""
        self.processed_image = None
        self.processing_history = []
        self.processing_times = {}
    
    def _add_to_history(self, operation, parameters):
        """Add operation to processing history"""
        history_entry = {
            'timestamp': datetime.now(),
            'operation': operation,
            'parameters': parameters,
            'processing_time': self.processing_times.get(operation, 0)
        }
        self.processing_history.append(history_entry)

class ModelManager:
    """
    Manager class for handling multiple AI models and processing queues
    """
    
    def __init__(self):
        """Initialize model manager"""
        self.models = {}
        self.active_model = None
        self.processing_queue = []
        
    def create_model(self, model_id="default"):
        """
        Create a new AI model instance
        
        Args:
            model_id: str - Unique identifier for the model
            
        Returns:
            AIModel: Created model instance
        """
        model = AIModel()
        self.models[model_id] = model
        if self.active_model is None:
            self.active_model = model_id
        return model
    
    def get_model(self, model_id="default"):
        """
        Get model by ID
        
        Args:
            model_id: str - Model identifier
            
        Returns:
            AIModel: Model instance or None
        """
        return self.models.get(model_id)
    
    def set_active_model(self, model_id):
        """
        Set active model
        
        Args:
            model_id: str - Model identifier
        """
        if model_id in self.models:
            self.active_model = model_id
    
    def get_active_model(self):
        """
        Get currently active model
        
        Returns:
            AIModel: Active model instance
        """
        return self.models.get(self.active_model)
    
    def list_models(self):
        """
        List all available models
        
        Returns:
            list: List of model IDs
        """
        return list(self.models.keys())
    
    def remove_model(self, model_id):
        """
        Remove model from manager
        
        Args:
            model_id: str - Model identifier
        """
        if model_id in self.models:
            del self.models[model_id]
            if self.active_model == model_id:
                self.active_model = list(self.models.keys())[0] if self.models else None

# Global model manager instance
model_manager = ModelManager()

def get_model(model_id="default"):
    """
    Get or create model instance
    
    Args:
        model_id: str - Model identifier
        
    Returns:
        AIModel: Model instance
    """
    model = model_manager.get_model(model_id)
    if model is None:
        model = model_manager.create_model(model_id)
    return model

def process_image_with_filters(image_array, filters_config):
    """
    Process image with multiple filters
    
    Args:
        image_array: numpy.ndarray - Input image
        filters_config: dict - Filters configuration
        
    Returns:
        tuple: (success: bool, result_image: numpy.ndarray, message: str)
    """
    model = get_model()
    model.current_image = image_array
    
    # Apply filters based on configuration
    if 'basic_filter' in filters_config and filters_config['basic_filter'] != "None":
        success, result, message = model.apply_basic_filter(
            filters_config['basic_filter'],
            **filters_config.get('basic_params', {})
        )
        if not success:
            return False, None, message
    
    if 'creative_filter' in filters_config and filters_config['creative_filter'] != "None":
        success, result, message = model.apply_creative_filter(
            filters_config['creative_filter'],
            **filters_config.get('creative_params', {})
        )
        if not success:
            return False, None, message
    
    if 'ai_filter' in filters_config and filters_config['ai_filter'] != "None":
        success, result, message = model.apply_ai_effect(
            filters_config['ai_filter'],
            **filters_config.get('ai_params', {})
        )
        if not success:
            return False, None, message
    
    return True, model.processed_image, "Processing completed successfully"

# Convenience functions for Streamlit integration
def create_processing_pipeline():
    """Create a processing pipeline for Streamlit"""
    return get_model()

def apply_filter_to_image(image_array, filter_type, filter_name, **parameters):
    """
    Apply filter to image (convenience function)
    
    Args:
        image_array: numpy.ndarray - Input image
        filter_type: str - Type of filter (basic, creative, ai)
        filter_name: str - Name of filter
        **parameters: Filter parameters
        
    Returns:
        tuple: (success: bool, result_image: numpy.ndarray, message: str)
    """
    model = get_model()
    model.current_image = image_array
    
    if filter_type == "basic":
        return model.apply_basic_filter(filter_name, **parameters)
    elif filter_type == "creative":
        return model.apply_creative_filter(filter_name, **parameters)
    elif filter_type == "ai":
        return model.apply_ai_effect(filter_name, **parameters)
    else:
        return False, None, f"Unknown filter type: {filter_type}"

if __name__ == "__main__":
    # Test the model
    print("AI Image Generator Model Module")
    print("Available functions:")
    print("- get_model(): Get or create model instance")
    print("- process_image_with_filters(): Process image with filters")
    print("- apply_filter_to_image(): Apply single filter")
    print("- create_processing_pipeline(): Create processing pipeline")
