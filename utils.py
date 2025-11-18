#!/usr/bin/env python3
"""
Utility Functions
Helper functions for the AI Image Generator project
"""

import os
import tempfile
import base64
from datetime import datetime
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def create_temp_file(suffix='.png'):
    """
    Create a temporary file for processing
    
    Args:
        suffix: str - File extension
        
    Returns:
        str: Path to temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()
    return temp_file.name

def save_uploaded_file(uploaded_file):
    """
    Save uploaded file to temporary location
    
    Args:
        uploaded_file: StreamlitUploadedFile - Uploaded file
        
    Returns:
        str: Path to saved file
    """
    try:
        # Create temporary file
        temp_path = create_temp_file(suffix=f'.{uploaded_file.name.split(".")[-1]}')
        
        # Save uploaded file
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return None

def image_to_bytes(image_array, format='PNG'):
    """
    Convert image array to bytes for download
    
    Args:
        image_array: numpy.ndarray - Image array
        format: str - Output format
        
    Returns:
        bytes: Image as bytes
    """
    try:
        # Convert numpy array to PIL Image
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            pil_image = Image.fromarray(image_array, 'RGB')
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            pil_image = Image.fromarray(image_array, 'RGBA')
        else:
            pil_image = Image.fromarray(image_array, 'L')
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format=format)
        img_buffer.seek(0)
        
        return img_buffer.getvalue()
    except Exception as e:
        st.error(f"Error converting image to bytes: {str(e)}")
        return None

def bytes_to_image(image_bytes):
    """
    Convert bytes to image array
    
    Args:
        image_bytes: bytes - Image data
        
    Returns:
        numpy.ndarray: Image array
    """
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        return image_array
    except Exception as e:
        st.error(f"Error converting bytes to image: {str(e)}")
        return None

def get_image_download_link(image_array, filename, text="Download Image"):
    """
    Generate download link for image
    
    Args:
        image_array: numpy.ndarray - Image to download
        filename: str - Download filename
        text: str - Link text
        
    Returns:
        str: HTML download link
    """
    try:
        # Convert image to bytes
        image_bytes = image_to_bytes(image_array)
        if image_bytes is None:
            return None
        
        # Encode to base64
        b64 = base64.b64encode(image_bytes).decode()
        
        # Get file extension
        file_extension = filename.split('.')[-1].lower()
        mime_type = f"image/{file_extension}"
        
        # Create download link
        href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{text}</a>'
        
        return href
    except Exception as e:
        st.error(f"Error generating download link: {str(e)}")
        return None

def validate_image(image_array):
    """
    Validate image array
    
    Args:
        image_array: numpy.ndarray - Image to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check if array is not None
        if image_array is None:
            return False
        
        # Check if array has valid shape
        if len(image_array.shape) < 2 or len(image_array.shape) > 3:
            return False
        
        # Check if array has valid dimensions
        if image_array.shape[0] <= 0 or image_array.shape[1] <= 0:
            return False
        
        # Check if array has valid data type
        if image_array.dtype not in [np.uint8, np.float32, np.float64]:
            return False
        
        return True
    except Exception:
        return False

def get_image_stats(image_array):
    """
    Get image statistics
    
    Args:
        image_array: numpy.ndarray - Input image
        
    Returns:
        dict: Image statistics
    """
    try:
        stats = {
            'shape': image_array.shape,
            'dtype': str(image_array.dtype),
            'min_value': float(np.min(image_array)),
            'max_value': float(np.max(image_array)),
            'mean_value': float(np.mean(image_array)),
            'std_value': float(np.std(image_array))
        }
        
        if len(image_array.shape) == 3:
            stats['channels'] = image_array.shape[2]
        else:
            stats['channels'] = 1
        
        return stats
    except Exception as e:
        return {'error': str(e)}

def create_comparison_image(original, processed):
    """
    Create side-by-side comparison image
    
    Args:
        original: numpy.ndarray - Original image
        processed: numpy.ndarray - Processed image
        
    Returns:
        numpy.ndarray: Comparison image
    """
    try:
        # Ensure both images have the same number of channels
        if len(original.shape) == 3 and len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        elif len(original.shape) == 2 and len(processed.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        
        # Resize images to same height
        height = min(original.shape[0], processed.shape[0])
        width1 = int(original.shape[1] * height / original.shape[0])
        width2 = int(processed.shape[1] * height / processed.shape[0])
        
        original_resized = cv2.resize(original, (width1, height))
        processed_resized = cv2.resize(processed, (width2, height))
        
        # Create comparison image
        comparison = np.hstack([original_resized, processed_resized])
        
        return comparison
    except Exception as e:
        st.error(f"Error creating comparison image: {str(e)}")
        return original

def apply_watermark(image_array, watermark_text="AI Generated"):
    """
    Apply watermark to image
    
    Args:
        image_array: numpy.ndarray - Input image
        watermark_text: str - Watermark text
        
    Returns:
        numpy.ndarray: Watermarked image
    """
    try:
        # Create a copy of the image
        result = image_array.copy()
        
        # Get image dimensions
        height, width = image_array.shape[:2]
        
        # Set font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(width, height) / 1000.0  # Scale font based on image size
        thickness = max(1, int(font_scale * 2))
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(watermark_text, font, font_scale, thickness)
        
        # Calculate position (bottom right corner)
        x = width - text_width - 20
        y = height - 20
        
        # Add semi-transparent background
        overlay = result.copy()
        cv2.rectangle(overlay, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)
        
        # Add text
        cv2.putText(result, watermark_text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        return result
    except Exception as e:
        st.error(f"Error applying watermark: {str(e)}")
        return image_array

def optimize_image_size(image_array, max_size=1024):
    """
    Optimize image size for processing
    
    Args:
        image_array: numpy.ndarray - Input image
        max_size: int - Maximum dimension size
        
    Returns:
        numpy.ndarray: Optimized image
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
        st.error(f"Error optimizing image size: {str(e)}")
        return image_array

def create_thumbnail(image_array, size=(200, 200)):
    """
    Create thumbnail of image
    
    Args:
        image_array: numpy.ndarray - Input image
        size: tuple - Thumbnail size (width, height)
        
    Returns:
        numpy.ndarray: Thumbnail image
    """
    try:
        # Resize image to thumbnail size
        thumbnail = cv2.resize(image_array, size, interpolation=cv2.INTER_AREA)
        
        return thumbnail
    except Exception as e:
        st.error(f"Error creating thumbnail: {str(e)}")
        return image_array

def get_file_size_mb(file_path):
    """
    Get file size in MB
    
    Args:
        file_path: str - Path to file
        
    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except Exception:
        return 0.0

def format_file_size(size_bytes):
    """
    Format file size in human readable format
    
    Args:
        size_bytes: int - Size in bytes
        
    Returns:
        str: Formatted size string
    """
    try:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    except Exception:
        return "Unknown size"

def create_progress_bar():
    """
    Create a progress bar for processing
    
    Returns:
        streamlit.progress: Progress bar object
    """
    return st.progress(0)

def show_processing_status(message):
    """
    Show processing status message
    
    Args:
        message: str - Status message
    """
    st.info(message)

def handle_errors(func):
    """
    Decorator to handle errors gracefully
    
    Args:
        func: function - Function to wrap
        
    Returns:
        function: Wrapped function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper
