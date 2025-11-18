#!/usr/bin/env python3
"""
Filters Module
Contains all image filtering and artistic effect functions
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from skimage import filters, feature, morphology
from skimage.color import rgb2gray, gray2rgb

class BasicFilters:
    """Basic image filters and effects"""
    
    @staticmethod
    def grayscale(image_array):
        """
        Convert image to grayscale
        
        Args:
            image_array: numpy.ndarray - Input image
            
        Returns:
            numpy.ndarray: Grayscale image
        """
        if len(image_array.shape) == 3:
            return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        return image_array
    @staticmethod
    def blur(image_array, kernel_size=5):
        """
        Apply Gaussian blur to image
        
        Args:
            image_array: numpy.ndarray - Input image
            kernel_size: int - Blur kernel size
            
        Returns:
            numpy.ndarray: Blurred image
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if len(image_array.shape) == 3:
            return cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)
        else:
            return cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def edge_detection(image_array, threshold1=50, threshold2=150):
        """
        Apply Canny edge detection
        
        Args:
            image_array: numpy.ndarray - Input image
            threshold1: int - First threshold for edge detection
            threshold2: int - Second threshold for edge detection
            
        Returns:
            numpy.ndarray: Edge detected image
        """
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges
    
    @staticmethod
    def sepia(image_array, intensity=0.8):
        """
        Apply sepia tone effect
        
        Args:
            image_array: numpy.ndarray - Input image
            intensity: float - Sepia intensity (0-1)
            
        Returns:
            numpy.ndarray: Sepia toned image
        """
        if len(image_array.shape) != 3:
            return image_array
        
        # Sepia transformation matrix
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # Apply sepia transformation
        sepia_image = cv2.transform(image_array, sepia_matrix)
        
        # Blend with original image based on intensity
        result = cv2.addWeighted(image_array, 1 - intensity, sepia_image, intensity, 0)
        
        # Ensure values are in valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def invert(image_array):
        """
        Invert image colors
        
        Args:
            image_array: numpy.ndarray - Input image
            
        Returns:
            numpy.ndarray: Inverted image
        """
        return 255 - image_array

class CreativeFilters:
    """Creative and artistic filters"""
    
    @staticmethod
    def cartoonify(image_array, num_bilateral=7, num_dilation=2):
        """
        Apply cartoon effect to image
        
        Args:
            image_array: numpy.ndarray - Input image
            num_bilateral: int - Number of bilateral filter passes
            num_dilation: int - Number of dilation operations
            
        Returns:
            numpy.ndarray: Cartoonified image
        """
        if len(image_array.shape) != 3:
            return image_array
        
        # Convert to RGB if needed
        if image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Apply bilateral filter multiple times
        color = image_array.copy()
        for _ in range(num_bilateral):
            color = cv2.bilateralFilter(color, 9, 9, 7)
        
        # Convert to grayscale
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        
        # Apply median blur
        gray = cv2.medianBlur(gray, 7)
        
        # Detect edges
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        
        # Dilate edges
        kernel = np.ones((2, 2), np.uint8)
        for _ in range(num_dilation):
            edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Combine color and edges
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        return cartoon
    
    @staticmethod
    def pencil_sketch(image_array, blur_factor=5):
        """
        Convert image to pencil sketch
        
        Args:
            image_array: numpy.ndarray - Input image
            blur_factor: int - Blur factor for sketch effect
            
        Returns:
            numpy.ndarray: Pencil sketch image
        """
        if len(image_array.shape) != 3:
            return image_array
        
        # Convert to RGB if needed
        if image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Invert the grayscale image
        inverted = 255 - gray
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted, (blur_factor, blur_factor), 0)
        
        # Blend the grayscale image with the blurred inverted image
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        
        return sketch
    
    @staticmethod
    def oil_painting(image_array, radius=4, intensity=25):
        """
        Apply oil painting effect
        
        Args:
            image_array: numpy.ndarray - Input image
            radius: int - Painting radius
            intensity: int - Painting intensity
            
        Returns:
            numpy.ndarray: Oil painting effect image
        """
        if len(image_array.shape) != 3:
            return image_array
        
        # Convert to RGB if needed
        if image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        try:
            # Try to apply oil painting effect using OpenCV xphoto module
            result = cv2.xphoto.oilPainting(image_array, radius, intensity)
            if result is not None:
                return result
        except:
            pass
        
        # Alternative implementation using bilateral filter and color quantization
        # Apply bilateral filter for edge-preserving smoothing
        result = cv2.bilateralFilter(image_array, 9, 75, 75)
        
        # Apply color quantization to simulate oil painting effect
        result = cv2.pyrMeanShiftFiltering(result, 21, 51)
        
        # Apply additional bilateral filter for more painting-like effect
        result = cv2.bilateralFilter(result, 9, 75, 75)
        
        return result
    
    @staticmethod
    def watercolor(image_array, blur_radius=3, edge_strength=0.5):
        """
        Apply watercolor effect
        
        Args:
            image_array: numpy.ndarray - Input image
            blur_radius: int - Blur radius for watercolor effect
            edge_strength: float - Edge strength (0-1)
            
        Returns:
            numpy.ndarray: Watercolor effect image
        """
        if len(image_array.shape) != 3:
            return image_array
        
        # Convert to RGB if needed
        if image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Apply bilateral filter for smoothing
        smoothed = cv2.bilateralFilter(image_array, 9, 75, 75)
        
        # Apply additional blur
        blurred = cv2.GaussianBlur(smoothed, (blur_radius, blur_radius), 0)
        
        # Detect edges
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Convert edges to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Blend blurred image with edges
        result = cv2.addWeighted(blurred, 1 - edge_strength, edges_rgb, edge_strength, 0)
        
        return result

class AIEffects:
    """Advanced AI-powered effects"""
    
    @staticmethod
    def neural_style_transfer(image_array, style_image_path=None):
        """
        Apply neural style transfer (placeholder for AI implementation)
        
        Args:
            image_array: numpy.ndarray - Input image
            style_image_path: str - Path to style image
            
        Returns:
            numpy.ndarray: Style transferred image
        """
        # Placeholder implementation - returns a stylized version using basic filters
        if len(image_array.shape) != 3:
            return image_array
        
        # Apply a combination of filters to simulate style transfer
        # Convert to HSV for color manipulation
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Enhance saturation
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)
        
        # Convert back to RGB
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Apply slight blur for artistic effect
        result = cv2.GaussianBlur(result, (3, 3), 0)
        
        return result
    
    @staticmethod
    def artistic_style(image_array, style_type="vangogh"):
        """
        Apply artistic style transformation
        
        Args:
            image_array: numpy.ndarray - Input image
            style_type: str - Style type (vangogh, picasso, etc.)
            
        Returns:
            numpy.ndarray: Artistically styled image
        """
        if len(image_array.shape) != 3:
            return image_array
        
        # Convert to RGB if needed
        if image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        if style_type == "vangogh":
            # Van Gogh style: vibrant colors, brush strokes
            # Enhance saturation and contrast
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.5)  # Increase saturation
            hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.2)  # Increase brightness
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Apply directional blur to simulate brush strokes
            kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
            result = cv2.filter2D(result, -1, kernel)
            
        elif style_type == "picasso":
            # Picasso style: cubist, geometric
            # Apply edge detection and color segmentation
            edges = cv2.Canny(image_array, 50, 150)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # Reduce colors
            result = cv2.pyrMeanShiftFiltering(image_array, 21, 51)
            
            # Combine with edges
            result = cv2.addWeighted(result, 0.7, edges_rgb, 0.3, 0)
            
        else:
            # Default artistic style
            try:
                result = cv2.stylization(image_array, sigma_s=60, sigma_r=0.4)
            except:
                # Fallback to basic artistic effect
                result = cv2.bilateralFilter(image_array, 9, 75, 75)
                # Enhance colors
                hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
                hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return result
    
    @staticmethod
    def photo_to_art(image_array, art_style="impressionist"):
        """
        Convert photo to artistic painting style
        
        Args:
            image_array: numpy.ndarray - Input image
            art_style: str - Art style (impressionist, abstract, etc.)
            
        Returns:
            numpy.ndarray: Artistic painting image
        """
        if len(image_array.shape) != 3:
            return image_array
        
        # Convert to RGB if needed
        if image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        if art_style == "impressionist":
            # Impressionist style: soft, colorful, brush-like
            # Apply bilateral filter for edge-preserving smoothing
            result = cv2.bilateralFilter(image_array, 9, 75, 75)
            
            # Enhance colors
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Apply slight blur for softness
            result = cv2.GaussianBlur(result, (3, 3), 0)
            
        elif art_style == "abstract":
            # Abstract style: geometric patterns, high contrast
            # Apply edge detection
            edges = cv2.Canny(image_array, 30, 100)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # Apply color quantization
            result = cv2.pyrMeanShiftFiltering(image_array, 21, 51)
            
            # Combine with edges
            result = cv2.addWeighted(result, 0.6, edges_rgb, 0.4, 0)
            
        else:
            # Default painting style
            try:
                result = cv2.stylization(image_array, sigma_s=60, sigma_r=0.4)
            except:
                # Fallback to basic painting effect
                result = cv2.bilateralFilter(image_array, 9, 75, 75)
                # Apply slight blur for softness
                result = cv2.GaussianBlur(result, (3, 3), 0)
        
        return result

class FilterProcessor:
    """Main filter processing class"""
    
    def __init__(self):
        """Initialize filter processor"""
        self.basic_filters = BasicFilters()
        self.creative_filters = CreativeFilters()
        self.ai_effects = AIEffects()
    
    def apply_filter(self, image_array, filter_name, **kwargs):
        """
        Apply specified filter to image
        
        Args:
            image_array: numpy.ndarray - Input image
            filter_name: str - Name of filter to apply
            **kwargs: Additional filter parameters
            
        Returns:
            numpy.ndarray: Filtered image
        """
        # Basic filters
        if filter_name == "grayscale":
            return self.basic_filters.grayscale(image_array)
        elif filter_name == "blur":
            kernel_size = kwargs.get('kernel_size', 5)
            return self.basic_filters.blur(image_array, kernel_size)
        elif filter_name == "edge_detection":
            threshold1 = kwargs.get('threshold1', 50)
            threshold2 = kwargs.get('threshold2', 150)
            return self.basic_filters.edge_detection(image_array, threshold1, threshold2)
        elif filter_name == "sepia":
            intensity = kwargs.get('intensity', 0.8)
            return self.basic_filters.sepia(image_array, intensity)
        elif filter_name == "invert":
            return self.basic_filters.invert(image_array)
        
        # Creative filters
        elif filter_name == "cartoonify":
            num_bilateral = kwargs.get('num_bilateral', 7)
            num_dilation = kwargs.get('num_dilation', 2)
            return self.creative_filters.cartoonify(image_array, num_bilateral, num_dilation)
        elif filter_name == "pencil_sketch":
            blur_factor = kwargs.get('blur_factor', 5)
            return self.creative_filters.pencil_sketch(image_array, blur_factor)
        elif filter_name == "oil_painting":
            radius = kwargs.get('radius', 4)
            intensity = kwargs.get('intensity', 25)
            return self.creative_filters.oil_painting(image_array, radius, intensity)
        elif filter_name == "watercolor":
            blur_radius = kwargs.get('blur_radius', 3)
            edge_strength = kwargs.get('edge_strength', 0.5)
            return self.creative_filters.watercolor(image_array, blur_radius, edge_strength)
        
        # AI effects
        elif filter_name == "neural_style_transfer":
            style_image_path = kwargs.get('style_image_path')
            return self.ai_effects.neural_style_transfer(image_array, style_image_path)
        elif filter_name == "artistic_style":
            style_type = kwargs.get('style_type', 'vangogh')
            return self.ai_effects.artistic_style(image_array, style_type)
        elif filter_name == "photo_to_art":
            art_style = kwargs.get('art_style', 'impressionist')
            return self.ai_effects.photo_to_art(image_array, art_style)
        
        else:
            # Return original image if filter not found
            return image_array
    
    def get_available_filters(self):
        """
        Get list of available filters
        
        Returns:
            dict: Dictionary of available filters by category
        """
        return {
            "Basic": ["grayscale", "blur", "edge_detection", "sepia", "invert"],
            "Creative": ["cartoonify", "pencil_sketch", "oil_painting", "watercolor"],
            "AI": ["neural_style_transfer", "artistic_style", "photo_to_art"]
        }
