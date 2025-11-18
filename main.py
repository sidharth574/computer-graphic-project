#- python -m streamlit run main.py
#!/usr/bin/env python3
"""
AI Image Generator - Main Application
College Project: Transform images into artistic styles using AI and computer vision
"""

import streamlit as st
from PIL import Image

# Import your custom modules
from m import get_model


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="AI Image Generator",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🎨 AI Image Generator")
    st.markdown("Transform your photos into stunning artworks using AI and computer vision!")
    
    # Initialize the AI model
    model = get_model()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📁 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to transform"
        )
        
        if uploaded_file is not None:
            st.success("✅ Image uploaded successfully!")
            
            # Display image info
            image = Image.open(uploaded_file)
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Format:** {image.format}")
            
            # Filter selection
            st.header("🎭 Choose Effect")
            
            # Basic filters
            st.subheader("Basic Filters")
            basic_filter = st.selectbox(
                "Select basic filter:",
                ["None", "Grayscale", "Blur", "Edge Detection", "Sepia", "Inverted"]
            )
            
            # Creative filters
            st.subheader("Creative Filters")
            creative_filter = st.selectbox(
                "Select creative filter:",
                ["None", "Cartoonify", "Pencil Sketch", "Oil Painting", "Watercolor"]
            )
            
            # Advanced AI effects
            st.subheader("AI Effects")
            ai_filter = st.selectbox(
                "Select AI effect:",
                ["None", "Neural Style Transfer", "Artistic Style", "Photo to Art"]
            )
            
            # Parameters
            st.header("⚙️ Parameters")
            
            # Intensity slider
            intensity = st.slider("Effect Intensity", 0.0, 1.0, 0.5, 0.1)
            
            # Additional parameters
            blur_strength = st.slider("Blur Strength", 1, 20, 5)
            edge_threshold = st.slider("Edge Threshold", 50, 200, 100)
            
            # Process button
            if st.button("🚀 Generate Artwork", type="primary"):
                st.info("Processing image... Please wait.")
                
                # Load and validate the uploaded image
                success, image_array, message = model.load_and_validate_image(uploaded_file)
                
                if not success:
                    st.error(f"Error loading image: {message}")
                else:
                    # Store the original image
                    original_image = image_array.copy()
                    
                    # Apply selected filters
                    processed_image = original_image
                    
                    # Apply basic filter
                    if basic_filter != "None":
                        filter_mapping = {
                            "Grayscale": "grayscale",
                            "Blur": "blur",
                            "Edge Detection": "edge_detection",
                            "Sepia": "sepia",
                            "Inverted": "invert"
                        }
                        
                        filter_name = filter_mapping.get(basic_filter)
                        if filter_name:
                            if filter_name == "blur":
                                success, processed_image, message = model.apply_basic_filter(
                                    filter_name, kernel_size=blur_strength
                                )
                            elif filter_name == "edge_detection":
                                success, processed_image, message = model.apply_basic_filter(
                                    filter_name, threshold1=edge_threshold//2, threshold2=edge_threshold
                                )
                            elif filter_name == "sepia":
                                success, processed_image, message = model.apply_basic_filter(
                                    filter_name, intensity=intensity
                                )
                            else:
                                success, processed_image, message = model.apply_basic_filter(filter_name)
                            
                            if not success:
                                st.error(f"Error applying {basic_filter}: {message}")
                                processed_image = original_image
                    
                    # Apply creative filter
                    if creative_filter != "None":
                        filter_mapping = {
                            "Cartoonify": "cartoonify",
                            "Pencil Sketch": "pencil_sketch",
                            "Oil Painting": "oil_painting",
                            "Watercolor": "watercolor"
                        }
                        
                        filter_name = filter_mapping.get(creative_filter)
                        if filter_name:
                            success, processed_image, message = model.apply_creative_filter(
                                filter_name
                            )
                            
                            if not success:
                                st.error(f"Error applying {creative_filter}: {message}")
                                processed_image = original_image
                    
                    # Apply AI effect
                    if ai_filter != "None":
                        filter_mapping = {
                            "Neural Style Transfer": "neural_style_transfer",
                            "Artistic Style": "artistic_style",
                            "Photo to Art": "photo_to_art"
                        }
                        
                        filter_name = filter_mapping.get(ai_filter)
                        if filter_name:
                            if filter_name == "artistic_style":
                                success, processed_image, message = model.apply_ai_effect(
                                    filter_name, style_type="vangogh"
                                )
                            elif filter_name == "photo_to_art":
                                success, processed_image, message = model.apply_ai_effect(
                                    filter_name, art_style="impressionist"
                                )
                            else:
                                success, processed_image, message = model.apply_ai_effect(filter_name)
                            
                            if not success:
                                st.error(f"Error applying {ai_filter}: {message}")
                                processed_image = original_image
                    
                    # Store the processed image in session state
                    if processed_image is not None:
                        st.session_state['processed_image'] = processed_image
                        st.session_state['original_image'] = original_image
                        st.success("✅ Image processed successfully!")
                    else:
                        st.error("❌ Failed to process image")
            
    # Main content area
    if 'processed_image' in st.session_state and 'original_image' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(st.session_state['original_image'], caption="Original", use_column_width=True)
            
        with col2:
            st.subheader("🎨 Generated Artwork")
            st.image(st.session_state['processed_image'], caption="Processed", use_column_width=True)
            
            # Download button
            if st.button("💾 Download Processed Image"):
                download_link = model.get_download_link("ai_generated_image.png")
                if download_link:
                    st.markdown(download_link, unsafe_allow_html=True)
                else:
                    st.error("Failed to generate download link")
            
            # Processing statistics
            if st.checkbox("📊 Show Processing Statistics"):
                stats = model.get_processing_stats()
                st.write("**Processing Statistics:**")
                st.write(f"- Total operations: {stats['total_operations']}")
                st.write(f"- Average processing time: {stats['average_time']:.3f} seconds")
                
                if stats['processing_times']:
                    st.write("**Operation Times:**")
                    for operation, time_taken in stats['processing_times'].items():
                        st.write(f"- {operation}: {time_taken:.3f} seconds")
    else:
        st.info("👆 Please upload an image from the sidebar to get started!")
        
        # Sample images or instructions
        st.markdown("""
        ### How to use:
        1. **Upload an image** using the file uploader in the sidebar
        2. **Choose your desired effect** from the filter options
        3. **Adjust parameters** to fine-tune the effect
        4. **Click 'Generate Artwork'** to see the transformation
        5. **Download** your masterpiece!
        """)
        
        # Features showcase
        st.markdown("""
        ### Available Features:
        - **Basic Filters**: Grayscale, Blur, Edge Detection, Sepia, Inverted
        - **Creative Filters**: Cartoonify, Pencil Sketch, Oil Painting, Watercolor
        - **AI Effects**: Neural Style Transfer, Artistic Styles
        - **Real-time Processing**: See results instantly
        - **Download Support**: Save your artwork
        """)


if __name__ == "__main__":
    main()
