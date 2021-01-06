from fastai.vision.all import *
import streamlit as st
import os
import time

# App title
st.title("PETS Classifier")

def predict(img, display_img):
    st.image(img, use_column_width=True)
    with st.spinner('Wait for it...'): time.sleep(3)

    # Load model and make prediction
    learn = load_learner('pets.pkl')
    clas, clas_idx, probs = learn.predict(img)
    prob = round(torch.max(probs).item() * 100, 2)
    
    # Display the prediction
    st.success(f"This is {clas} with the proability of {prob}%.")


# Test image selection
test_images = os.listdir('images/')
test_image = st.selectbox('Please select a test image:', test_images)

# Read and predict the image
file_path = 'images/' + test_image
img = PILImage.create(file_path)
predict(img)