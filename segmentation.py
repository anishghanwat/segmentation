import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_image(uploaded_file):
    if uploaded_file is not None:
        try:
            # Read file as bytes
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            # Decode the image
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Failed to decode image. Please try another file.")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None
    return None

def threshold_segmentation(img):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    except Exception as e:
        st.error(f"Error in threshold segmentation: {str(e)}")
        return None

def kmeans_segmentation(img, k=3):
    try:
        # Reshape the image
        pixel_values = img.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        
        # Reshape back to the original image dimensions
        segmented_image = segmented_data.reshape(img.shape)
        return segmented_image
    except Exception as e:
        st.error(f"Error in k-means segmentation: {str(e)}")
        return None

def main():
    st.title("Image Segmentation App")
    st.write("Upload an image and choose a segmentation method to analyze it.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = load_image(uploaded_file)
        
        if image is not None:
            st.image(image, caption='Original Image', use_column_width=True)
            
            # Segmentation method selection
            method = st.selectbox(
                "Choose segmentation method",
                ["Threshold", "K-means"]
            )
            
            # Parameters based on method
            if method == "K-means":
                k_clusters = st.slider("Number of clusters", 2, 8, 3)
            
            # Process button
            if st.button("Segment Image"):
                with st.spinner('Processing...'):
                    try:
                        # Apply selected segmentation
                        if method == "Threshold":
                            result = threshold_segmentation(image)
                        else:  # K-means
                            result = kmeans_segmentation(image, k_clusters)
                        
                        if result is not None:
                            st.success("Segmentation completed!")
                            st.image(result, caption='Segmented Image', use_column_width=True)
                            
                            # Add download button
                            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                            if is_success:
                                btn = st.download_button(
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