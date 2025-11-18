#!/usr/bin/env python3
"""
Test Application
Simple test to verify all dependencies are working correctly
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from skimage import filters, feature
import os

def test_dependencies():
    """Test if all dependencies are working"""
    
    st.title("🧪 Dependency Test")
    st.write("Testing if all required packages are working correctly...")
    
    # Test OpenCV
    try:
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # Red channel
        
        # Apply a simple OpenCV operation
        gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        st.success("✅ OpenCV is working!")
    except Exception as e:
        st.error(f"❌ OpenCV error: {e}")
    
    # Test NumPy
    try:
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.sum(arr)
        st.success("✅ NumPy is working!")
    except Exception as e:
        st.error(f"❌ NumPy error: {e}")
    
    # Test PIL
    try:
        img = Image.new('RGB', (100, 100), color='red')
        st.success("✅ PIL/Pillow is working!")
    except Exception as e:
        st.error(f"❌ PIL error: {e}")
    
    # Test Matplotlib
    try:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        st.success("✅ Matplotlib is working!")
    except Exception as e:
        st.error(f"❌ Matplotlib error: {e}")
    
    # Test Seaborn
    try:
        data = np.random.randn(100)
        fig, ax = plt.subplots()
        sns.histplot(data, ax=ax)
        st.success("✅ Seaborn is working!")
    except Exception as e:
        st.error(f"❌ Seaborn error: {e}")
    
    # Test Plotly
    try:
        df = {'x': [1, 2, 3], 'y': [1, 4, 2]}
        fig = px.line(df, x='x', y='y')
        st.success("✅ Plotly is working!")
    except Exception as e:
        st.error(f"❌ Plotly error: {e}")
    
    # Test scikit-image
    try:
        test_img = np.random.rand(100, 100)
        edges = filters.sobel(test_img)
        st.success("✅ scikit-image is working!")
    except Exception as e:
        st.error(f"❌ scikit-image error: {e}")
    
    # Test Streamlit
    try:
        st.write("✅ Streamlit is working!")
        st.balloons()
    except Exception as e:
        st.error(f"❌ Streamlit error: {e}")

def show_system_info():
    """Display system information"""
    
    st.header("💻 System Information")
    
    # Python version
    import sys
    st.write(f"**Python Version:** {sys.version}")
    
    # Package versions
    st.write("**Package Versions:**")
    packages = [
        ("OpenCV", cv2.__version__),
        ("NumPy", np.__version__),
        ("PIL", Image.__version__),
        ("Matplotlib", plt.matplotlib.__version__),
        ("Seaborn", sns.__version__),
        ("Plotly", px.__version__),
        ("Streamlit", st.__version__)
    ]
    
    for package, version in packages:
        st.write(f"- {package}: {version}")

def create_sample_image():
    """Create a sample image for testing"""
    
    st.header("🎨 Sample Image Creation")
    
    # Create a colorful test image
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add some colorful shapes
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(img, (250, 100), 50, (0, 255, 0), -1)  # Green circle
    cv2.line(img, (100, 200), (300, 250), (0, 0, 255), 5)  # Red line
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    st.write("**Sample Test Image:**")
    st.image(img_rgb, caption="Test Image with Shapes", use_column_width=True)
    
    return img_rgb

def main():
    """Main test function"""
    
    st.set_page_config(
        page_title="AI Image Generator - Test",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 AI Image Generator - Dependency Test")
    st.write("This page tests if all required dependencies are working correctly.")
    
    # Run tests
    test_dependencies()
    
    # Show system info
    show_system_info()
    
    # Create sample image
    sample_img = create_sample_image()
    
    st.success("🎉 All tests completed! If you see all green checkmarks, your setup is ready!")
    st.info("You can now start implementing your AI Image Generator features!")

if __name__ == "__main__":
    main()
