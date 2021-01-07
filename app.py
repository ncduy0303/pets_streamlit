from fastai.vision.all import load_learner, torch, PILImage, download_url
import streamlit as st
import os
import time

st.title("PETS Classifier")

download_url('https://math-doodle-models.s3-ap-southeast-1.amazonaws.com/pets.pkl', './model.pkl')

def predict(img):
    st.image(img, use_column_width=True)
    with st.spinner('Wait for it...'): time.sleep(3)

    learn = load_learner('model.pkl')
    clas, clas_idx, probs = learn.predict(img)
    prob = round(torch.max(probs).item() * 100, 2)
    
    st.success(f"This is {clas} with the proability of {prob}%.")


option = st.radio('', ['Choose a test image', 'Choose your own image'])

if option == 'Choose a test image':
    test_images = os.listdir('images/')
    test_image = st.selectbox('Please select a test image:', test_images)
    file_path = 'images/' + test_image
    img = PILImage.create(file_path)
    predict(img)

else:
    uploaded_file = st.file_uploader("Please upload an image", type="jpg")
    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
        predict(img)