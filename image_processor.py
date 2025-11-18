#!/usr/bin/env python3
"""
Image Processor Module
Handles core image processing operations for the AI Image Generator
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64
from PIL import ImageEnhance

class ImageProcessor:
    """Main image processing class"""
    
    def __init__(self):
        """Initialize the image processor"""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
    def load_image(self, image_file):
        """
        Load image from file upload or path
        
        Args:
            image_file: Streamlit uploaded file or file path
            
        Returns:
            numpy.ndarray: Loaded image as numpy array
        """
        try:
            if hasattr(image_file, 'read'):  # StreamlitUploadedFile
                # Read the uploaded file
                image_bytes = image_file.read()
                image_file.seek(0)  # Reset file pointer
                
                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to numpy array
                image_array = np.array(pil_image)
                
            else:  # File path
                # Load using OpenCV
                image_array = cv2.imread(image_file)
                if image_array is None:
                    raise ValueError(f"Could not load image from {image_file}")
                
                # Convert BGR to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
    
    def save_image(self, image_array, format='PNG'):
        """
        Save image array to bytes for download
        
        Args:
            image_array: numpy.ndarray - Image to save
            format: str - Output format (PNG, JPEG, etc.)
            
        Returns:
            bytes: Image data as bytes
        """
        try:
            # Convert numpy array to PIL Image
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                pil_image = Image.fromarray(image_array, 'RGB')
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                pil_image = Image.fromarray(image_array, 'RGBA')
            else:
                pil_image = Image.fromarray(image_array, 'L')
            
            # Save to bytes
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format=format)
            img_buffer.seek(0)
            
            return img_buffer.getvalue()
            
        except Exception as e:
            raise ValueError(f"Error saving image: {str(e)}")
    
    def resize_image(self, image_array, max_size=800):
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image_array: numpy.ndarray - Input image
            max_size: int - Maximum dimension size
            
        Returns:
            numpy.ndarray: Resized image
        """
        try:
            height, width = image_array.shape[:2]
            
            # Check if resizing is needed
            if height <= max_size and width <= max_size:
                return image_array
            
            # Calculate new dimensions
            if height > width:
                new_height = max_size
                new_width = int(width * max_size / height)
            else:
                new_width = max_size
                new_height = int(height * max_size / width)
            
            # Resize image
            resized = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return resized
            
        except Exception as e:
            raise ValueError(f"Error resizing image: {str(e)}")
    
    def convert_to_rgb(self, image_array):
        """
        Convert BGR image to RGB
        
        Args:
            image_array: numpy.ndarray - Input image (BGR format)
            
        Returns:
            numpy.ndarray: RGB image
        """
        try:
            if len(image_array.shape) == 3:
                return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            return image_array
            
        except Exception as e:
            raise ValueError(f"Error converting to RGB: {str(e)}")
    
    def get_image_info(self, image_array):
        """
        Get basic information about the image
        
        Args:
            image_array: numpy.ndarray - Input image
            
        Returns:
            dict: Image information (size, channels, dtype)
        """
        try:
            info = {
                'shape': image_array.shape,
                'height': image_array.shape[0],
                'width': image_array.shape[1],
                'dtype': str(image_array.dtype),
                'min_value': float(np.min(image_array)),
                'max_value': float(np.max(image_array)),
                'mean_value': float(np.mean(image_array))
            }
            
            if len(image_array.shape) == 3:
                info['channels'] = image_array.shape[2]
                info['color_mode'] = 'RGB' if image_array.shape[2] == 3 else 'RGBA'
            else:
                info['channels'] = 1
                info['color_mode'] = 'Grayscale'
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def preprocess_image(self, image_array):
        """
        Preprocess image for better processing results
        
        Args:
            image_array: numpy.ndarray - Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Create a copy to avoid modifying original
            processed = image_array.copy()
            
            # Ensure image is in RGB format
            if len(processed.shape) == 3 and processed.shape[2] == 4:
                # Convert RGBA to RGB
                processed = cv2.cvtColor(processed, cv2.COLOR_RGBA2RGB)
            
            # Normalize pixel values if needed
            if processed.dtype != np.uint8:
                processed = np.clip(processed, 0, 255).astype(np.uint8)
            
            # Apply slight sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            processed = cv2.filter2D(processed, -1, kernel)
            
            return processed
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def postprocess_image(self, image_array):
        """
        Postprocess image for better display
        
        Args:
            image_array: numpy.ndarray - Input image
            
        Returns:
            numpy.ndarray: Postprocessed image
        """
        try:
            # Create a copy to avoid modifying original
            processed = image_array.copy()
            
            # Ensure values are in valid range
            processed = np.clip(processed, 0, 255).astype(np.uint8)
            
            # Apply slight contrast enhancement
            if len(processed.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                
                # Convert back to RGB
                processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Apply CLAHE to grayscale image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(processed)
            
            return processed
            
        except Exception as e:
            raise ValueError(f"Error postprocessing image: {str(e)}")
    
    def enhance_image(self, image_array, brightness=1.0, contrast=1.0, saturation=1.0):
        """
        Enhance image with brightness, contrast, and saturation adjustments
        
        Args:
            image_array: numpy.ndarray - Input image
            brightness: float - Brightness multiplier
            contrast: float - Contrast multiplier
            saturation: float - Saturation multiplier
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        try:
            # Convert to PIL Image for enhancement
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                pil_image = Image.fromarray(image_array, 'RGB')
            else:
                pil_image = Image.fromarray(image_array, 'L')
            
            # Apply brightness enhancement
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(brightness)
            
            # Apply contrast enhancement
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(contrast)
            
            # Apply color enhancement (only for RGB images)
            if saturation != 1.0 and len(image_array.shape) == 3:
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(saturation)
            
            # Convert back to numpy array
            enhanced = np.array(pil_image)
            
            return enhanced
            
        except Exception as e:
            raise ValueError(f"Error enhancing image: {str(e)}")
    
    def apply_noise_reduction(self, image_array, method='bilateral'):
        """
        Apply noise reduction to image
        
        Args:
            image_array: numpy.ndarray - Input image
            method: str - Noise reduction method ('bilateral', 'gaussian', 'median')
            
        Returns:
            numpy.ndarray: Denoised image
        """
        try:
            if method == 'bilateral':
                # Bilateral filter preserves edges while reducing noise
                denoised = cv2.bilateralFilter(image_array, 9, 75, 75)
            elif method == 'gaussian':
                # Gaussian blur
                denoised = cv2.GaussianBlur(image_array, (5, 5), 0)
            elif method == 'median':
                # Median filter
                denoised = cv2.medianBlur(image_array, 5)
            else:
                raise ValueError(f"Unknown noise reduction method: {method}")
            
            return denoised
            
        except Exception as e:
            raise ValueError(f"Error applying noise reduction: {str(e)}")
    
    def convert_to_grayscale(self, image_array):
        """
        Convert image to grayscale
        
        Args:
            image_array: numpy.ndarray - Input image
            
        Returns:
            numpy.ndarray: Grayscale image
        """
        try:
            if len(image_array.shape) == 3:
                return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            return image_array
            
        except Exception as e:
            raise ValueError(f"Error converting to grayscale: {str(e)}")
    
    def apply_color_correction(self, image_array, temperature=0, tint=0):
        """
        Apply color correction (temperature and tint)
        
        Args:
            image_array: numpy.ndarray - Input image
            temperature: float - Color temperature adjustment (-100 to 100)
            tint: float - Color tint adjustment (-100 to 100)
            
        Returns:
            numpy.ndarray: Color corrected image
        """
        try:
            if len(image_array.shape) != 3:
                return image_array
            
            # Convert to LAB color space
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            
            # Apply temperature adjustment (affects L channel)
            if temperature != 0:
                lab[:, :, 0] = np.clip(lab[:, :, 0] + temperature, 0, 255)
            
            # Apply tint adjustment (affects a and b channels)
            if tint != 0:
                lab[:, :, 1] = np.clip(lab[:, :, 1] + tint, 0, 255)
                lab[:, :, 2] = np.clip(lab[:, :, 2] - tint, 0, 255)
            
            # Convert back to RGB
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return corrected
            
        except Exception as e:
            raise ValueError(f"Error applying color correction: {str(e)}")
