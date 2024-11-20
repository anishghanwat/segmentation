import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Image Segmentation App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Segmentation functions
def threshold_segmentation(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        st.error(f"Error in threshold segmentation: {str(e)}")
        return None

def kmeans_segmentation(img, k=3):
    try:
        pixel_values = img.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(img.shape)
        return segmented_image
    except Exception as e:
        st.error(f"Error in k-means segmentation: {str(e)}")
        return None

def watershed_segmentation(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Noise removal first
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(img.copy(), markers)
        img_result = img.copy()
        img_result[markers == -1] = [0, 0, 255]  # Mark boundaries in red
        
        return img_result
    except Exception as e:
        st.error(f"Error in watershed segmentation: {str(e)}")
        return None

def contour_segmentation(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output = np.zeros_like(img)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
        
        return output
    except Exception as e:
        st.error(f"Error in contour segmentation: {str(e)}")
        return None

def main():
    try:
        # Add custom CSS
        st.markdown("""
            <style>
                .stApp {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .upload-text {
                    font-size: 20px;
                    font-weight: bold;
                    margin-bottom: 20px;
                }
            </style>
        """, unsafe_allow_html=True)

        # App title and description
        st.title("Image Segmentation App")
        st.markdown("Upload an image and choose a segmentation method to analyze it.")

        # Create two columns
        col1, col2 = st.columns(2)

        with col1:
            # File uploader
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                try:
                    # Read image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Display original image
                    st.subheader("Original Image")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                    # Segmentation method selection
                    method = st.selectbox(
                        "Choose segmentation method",
                        ["Threshold", "K-means", "Watershed", "Contour"]
                    )

                    # Parameters for K-means
                    k_clusters = 3
                    if method == "K-means":
                        k_clusters = st.slider("Number of clusters", 2, 10, 3)

                    if st.button("Segment Image"):
                        with st.spinner('Processing...'):
                            # Apply selected segmentation method
                            if method == "Threshold":
                                result = threshold_segmentation(img)
                            elif method == "K-means":
                                result = kmeans_segmentation(img, k_clusters)
                            elif method == "Watershed":
                                result = watershed_segmentation(img)
                            else:  # Contour
                                result = contour_segmentation(img)

                            if result is not None:
                                with col2:
                                    st.subheader("Segmented Image")
                                    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

                                    # Add download button
                                    is_success, buffer = cv2.imencode(".png", result)
                                    if is_success:
                                        btn = st.download_button(
                                            label="Download segmented image",
                                            data=buffer.tobytes(),
                                            file_name="segmented_image.png",
                                            mime="image/png"
                                        )
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
            else:
                st.info("Please upload an image to begin")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
