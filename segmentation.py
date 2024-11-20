import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Add caching to improve performance
@st.cache_data
def load_image(uploaded_file):
    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Failed to decode image. Please try another file.")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None
    return None

@st.cache_data
def threshold_segmentation(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    except Exception as e:
        st.error(f"Error in threshold segmentation: {str(e)}")
        return None

@st.cache_data
def kmeans_segmentation(img, k=3):
    try:
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()].reshape(img.shape)
        return segmented
    except Exception as e:
        st.error(f"Error in k-means segmentation: {str(e)}")
        return None

@st.cache_data
def watershed_segmentation(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [255,0,0]
        
        return img
    except Exception as e:
        st.error(f"Error in watershed segmentation: {str(e)}")
        return None

@st.cache_data
def contour_segmentation(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        result = img.copy()
        cv2.drawContours(result, contours, -1, (0,255,0), 2)
        return result
    except Exception as e:
        st.error(f"Error in contour segmentation: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Image Segmentation App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Image Segmentation App")
    st.write("Upload an image and choose a segmentation method to analyze it.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            if image is not None:
                st.image(image, caption='Original Image', use_column_width=True)
    
    with col2:
        if uploaded_file is not None and image is not None:
            method = st.selectbox(
                "Choose segmentation method",
                ["Threshold", "K-means", "Watershed", "Contour"]
            )
            
            # Parameters based on method
            params = {}
            if method == "K-means":
                params['k_clusters'] = st.slider("Number of clusters", 2, 8, 3)
            
            if st.button("Segment Image", type="primary"):
                with st.spinner('Processing...'):
                    try:
                        # Apply selected segmentation
                        if method == "Threshold":
                            result = threshold_segmentation(image)
                        elif method == "K-means":
                            result = kmeans_segmentation(image, params['k_clusters'])
                        elif method == "Watershed":
                            result = watershed_segmentation(image)
                        else:  # Contour
                            result = contour_segmentation(image)
                        
                        if result is not None:
                            st.success("Segmentation completed!")
                            st.image(result, caption='Segmented Image', use_column_width=True)
                            
                            # Add download button
                            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                            if is_success:
                                st.download_button(
                                    label="Download segmented image",
                                    data=buffer.tobytes(),
                                    file_name="segmented_image.png",
                                    mime="image/png"
                                )
                        else:
                            st.error("Segmentation failed. Please try another image or method.")
                    except Exception as e:
                        st.error(f"Error during segmentation: {str(e)}")
        else:
            st.info("Please upload an image to begin")

if __name__ == "__main__":
    main()