import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64

# Load the YOLO model (make sure to use your own model path)
model = YOLO('yolov9c.pt')  # Path to your YOLOv9 model weights
st.set_page_config(page_title="Object Detection",layout="centered",page_icon=":eyes:")


st.markdown("""
<style>
.css-9s5bis.edgvbvh3{
    visibility :hidden;
}
.css-1q1n0ol.egzxvld0{
    visibility : hidden;
}
</style>
""",unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('b1.jpeg') 
# Streamlit app title
st.title("Object Detection with YOLOv9")

# Layout settings for a responsive design
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            font-family: Arial, sans-serif;
        }
        .stImage {
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton {
            font-size: 18px;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for user instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
    1. Upload an image for object detection.
    2. Click on "Detect Objects" to see the results.
    3. The detected objects will be shown on the image.
""")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

# Run the object detection when the user uploads an image
if uploaded_file is not None:
    try:
        # Open the image using PIL
        img = Image.open(uploaded_file)

        # Show the uploaded image in Streamlit with custom styling using markdown
        st.markdown('<div class="stImage"></div>', unsafe_allow_html=True)  # Apply custom styling to the image
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Convert the uploaded image to a format YOLO can work with (numpy array)
        img_array = np.array(img)

        # Check if the image has multiple channels (e.g., RGB or RGBA)
        if len(img_array.shape) < 3 or img_array.shape[2] not in [3, 4]:
            raise ValueError("Invalid image format or unsupported number of channels. Please upload a valid image.")

        # Perform object detection
        result = model.predict(img_array)

        # Display the results on the same image
        output_image = result[0].plot()  # This adds bounding boxes to the image

        # Show the image with detected objects
        st.image(output_image, caption="Detected Objects", use_column_width=True)

        # Convert the output image to a byte stream for download
        output_image_pil = Image.fromarray(output_image)
        img_byte_arr = io.BytesIO()
        output_image_pil.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # Provide a download button
        st.download_button(
            label="Download Detected Image",
            data=img_byte_arr,
            file_name="output_detected_image.png",
            mime="image/png"
        )

    except ValueError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
