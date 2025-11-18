# 🎨 AI Image Generator

A comprehensive AI-powered image transformation tool that converts ordinary photos into stunning artworks using computer vision and machine learning techniques.

## 📋 Project Overview

This AI art generator transforms ordinary images into visually creative artworks by applying artistic effects such as cartoonification, pencil sketching, or even replicating the styles of famous paintings through neural style transfer. The main goal is to provide users with an easy-to-use, interactive platform where they can upload photos and instantly see these artistic transformations.

## ✨ Features

### 🎭 Basic Filters
- **Grayscale**: Convert images to black and white
- **Blur**: Apply Gaussian blur effects
- **Edge Detection**: Highlight edges using Canny algorithm
- **Sepia**: Apply vintage sepia tone effects
- **Inverted**: Invert image colors

### 🎨 Creative Filters
- **Cartoonify**: Transform photos into cartoon-style artwork
- **Pencil Sketch**: Convert images to realistic pencil sketches
- **Oil Painting**: Apply oil painting artistic effects
- **Watercolor**: Create watercolor painting effects

### 🤖 AI Effects
- **Neural Style Transfer**: Apply artistic styles from famous paintings
- **Artistic Style**: Transform photos into various art styles
- **Photo to Art**: Convert photos to artistic paintings

### 🌐 Web Interface
- **Streamlit-based UI**: User-friendly web interface
- **Real-time Processing**: Instant visual feedback
- **Side-by-side Comparison**: View original and processed images
- **Download Support**: Save your artwork
- **Parameter Control**: Fine-tune effects with sliders

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project files**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run main.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
ai-image-generator/
├── main.py                 # Main Streamlit application
├── image_processor.py      # Core image processing functions
├── filters.py             # Basic and advanced filters
├── utils.py               # Utility functions
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── assets/               # Static assets (images, icons)
├── temp/                 # Temporary files
├── models/               # AI models (if using neural networks)
└── styles/               # Style images for neural transfer
```

## 🎯 Usage Guide

### Getting Started
1. **Upload an Image**: Use the file uploader in the sidebar
2. **Choose Effects**: Select from Basic, Creative, or AI filters
3. **Adjust Parameters**: Fine-tune effects using the sliders
4. **Generate Artwork**: Click the "Generate Artwork" button
5. **Download Result**: Save your masterpiece

### Supported File Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

### Image Requirements
- Maximum file size: 50MB
- Maximum dimensions: 2048x2048 pixels
- Supported color modes: RGB, RGBA

## 🔧 Technical Details

### Core Technologies
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision and image processing
- **Pillow (PIL)**: Image manipulation
- **NumPy**: Numerical computing
- **scikit-image**: Advanced image processing
- **Matplotlib**: Plotting and visualization

### Processing Pipeline
1. **Image Upload**: Validate and load uploaded image
2. **Preprocessing**: Resize and optimize for processing
3. **Filter Application**: Apply selected artistic effects
4. **Postprocessing**: Enhance and prepare for display
5. **Output**: Display and provide download option

### Performance Optimizations
- Image size optimization for faster processing
- Caching for repeated operations
- Memory management for large images
- Progress indicators for long operations

## 🎨 Filter Details

### Basic Filters Implementation
- **Grayscale**: RGB to grayscale conversion
- **Blur**: Gaussian blur with configurable kernel size
- **Edge Detection**: Canny edge detection with dual thresholds
- **Sepia**: Color matrix transformation
- **Invert**: 255 - pixel values

### Creative Filters Implementation
- **Cartoonify**: Bilateral filtering + edge detection + color quantization
- **Pencil Sketch**: Grayscale + Gaussian blur + edge detection
- **Oil Painting**: Color clustering + edge preservation
- **Watercolor**: Bilateral filtering + edge enhancement

### AI Effects (Advanced)
- **Neural Style Transfer**: Deep learning-based style transfer
- **Artistic Style**: Pre-trained model applications
- **Photo to Art**: CNN-based artistic transformation

## 🔒 Security Features

- File type validation
- File size limits
- Secure file handling
- Input sanitization
- Error handling and logging

## 📊 Performance Metrics

- Processing time: < 30 seconds for most effects
- Memory usage: Optimized for large images
- CPU utilization: Efficient algorithms
- GPU support: Optional CUDA acceleration

## 🐛 Troubleshooting

### Common Issues

1. **Installation Problems**
   - Ensure Python 3.8+ is installed
   - Update pip: `pip install --upgrade pip`
   - Install dependencies: `pip install -r requirements.txt`

2. **Runtime Errors**
   - Check file permissions
   - Verify image format support
   - Ensure sufficient disk space

3. **Performance Issues**
   - Reduce image size before upload
   - Close other applications
   - Check available memory

### Error Messages
- **"File too large"**: Reduce image size
- **"Unsupported format"**: Convert to JPEG or PNG
- **"Processing timeout"**: Try smaller image or simpler effect

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints
- Write unit tests

## 📝 License

This project is created for educational purposes as a college project.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Streamlit team for the web framework
- Python community for excellent libraries
- Academic resources for algorithm implementations

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review the code comments
- Consult the documentation
- Contact the development team

---

**Happy Creating! 🎨✨**
