#!/usr/bin/env python3
"""
Configuration Settings
Configuration file for the AI Image Generator project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
ASSETS_DIR = PROJECT_ROOT / "assets"
TEMP_DIR = PROJECT_ROOT / "temp"
MODELS_DIR = PROJECT_ROOT / "models"
STYLES_DIR = PROJECT_ROOT / "styles"

# Create directories if they don't exist
for directory in [ASSETS_DIR, TEMP_DIR, MODELS_DIR, STYLES_DIR]:
    directory.mkdir(exist_ok=True)

# Image processing settings
MAX_IMAGE_SIZE = 2048  # Maximum image dimension
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB

# Filter settings
DEFAULT_BLUR_KERNEL = 5
DEFAULT_EDGE_THRESHOLD1 = 50
DEFAULT_EDGE_THRESHOLD2 = 150
DEFAULT_SEPIA_INTENSITY = 0.8

# Creative filter settings
DEFAULT_CARTOON_BILATERAL = 7
DEFAULT_CARTOON_DILATION = 2
DEFAULT_SKETCH_BLUR = 5
DEFAULT_OIL_RADIUS = 4
DEFAULT_OIL_INTENSITY = 25
DEFAULT_WATERCOLOR_BLUR = 3
DEFAULT_WATERCOLOR_EDGE = 0.5

# AI settings
NEURAL_STYLE_MODELS = {
    "vangogh": "models/vangogh_style.pth",
    "picasso": "models/picasso_style.pth",
    "monet": "models/monet_style.pth"
}

# UI settings
SIDEBAR_WIDTH = 300
MAIN_WIDTH = 800
THUMBNAIL_SIZE = (200, 200)

# Processing settings
ENABLE_PROGRESS_BAR = True
ENABLE_WATERMARK = True
WATERMARK_TEXT = "AI Generated"

# Error handling
MAX_PROCESSING_TIME = 300  # seconds
RETRY_ATTEMPTS = 3

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = "ai_image_generator.log"

# Development settings
DEBUG_MODE = True
SHOW_PROCESSING_STEPS = True

# Performance settings
USE_GPU = False  # Set to True if CUDA is available
BATCH_SIZE = 1
MEMORY_LIMIT_MB = 2048

# Export settings
DEFAULT_EXPORT_FORMAT = "PNG"
EXPORT_QUALITY = 95  # For JPEG
ENABLE_METADATA = True

# Security settings
ALLOWED_FILE_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']
MAX_UPLOADS_PER_SESSION = 10

# Cache settings
ENABLE_CACHE = True
CACHE_TTL = 3600  # seconds
MAX_CACHE_SIZE_MB = 100

# API settings (if using external APIs)
API_TIMEOUT = 30
API_RETRY_DELAY = 5

# Style transfer settings
STYLE_WEIGHT = 1.0
CONTENT_WEIGHT = 1.0
ITERATIONS = 1000
LEARNING_RATE = 0.01

# Color settings
DEFAULT_COLOR_MODE = "RGB"
GRAYSCALE_MODE = "L"
RGBA_MODE = "RGBA"

# Filter presets
FILTER_PRESETS = {
    "vintage": {
        "sepia": 0.8,
        "blur": 2,
        "brightness": 0.9
    },
    "dramatic": {
        "contrast": 1.3,
        "saturation": 1.2,
        "brightness": 0.8
    },
    "soft": {
        "blur": 3,
        "brightness": 1.1,
        "saturation": 0.8
    }
}

# Default parameters
DEFAULT_PARAMETERS = {
    "intensity": 0.5,
    "blur_strength": 5,
    "edge_threshold": 100,
    "cartoon_level": 7,
    "sketch_detail": 5,
    "painting_radius": 4,
    "watercolor_blur": 3
}
