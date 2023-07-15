import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from CatsVsDogsNet import Net
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

model = Net()
model.load_state_dict(torch.load(DIR_PATH+"/model.pth", map_location=torch.device("cpu")))
model.eval()

def preprocess_image(image):
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformation to the image
    preprocessed_image = transform(image).unsqueeze(0)

    return preprocessed_image

def main():
    # st.title("Image Uploader")
    st.markdown("<h1 style='text-align: center;'>Is it a Cat or a Dog?</h1>", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="fileUploader", label_visibility="collapsed", help="Upload an image of a cat or a dog")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        image_height = 400
        image_width = int(image_height * image.width / image.height)
        image = image.resize((image_width, image_height))
        # st.image(image, caption=None)
        st.markdown(
            f"<div style='display: flex; justify-content: center;'>"
            f"<img src='data:image/jpeg;base64,{image_to_base64(image)}' alt='Uploaded Image' width={image_width}>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Process the image using PyTorch
        preprocessed_image = preprocess_image(image)
        output = torch.softmax(model(preprocessed_image), dim=1)
        cat_percentage = output[0][0].item() * 100
        dog_percentage = output[0][1].item() * 100
        pred = output.argmax(dim=1).item()
        label = ["Cat", "Dog"]
        color = ["#66B2FF", "#FF6666"]
        # Display centered text below the image
        st.markdown(f"<h1 style='text-align: center;'>It's a <span style='color: {color[pred]}'>{label[pred]}</span></h1>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center;'>Confidence: <span style='color: {color[0]}'>{cat_percentage:.1f}% cat</span> | <span style='color: {color[1]}'>{dog_percentage:.1f}% dog</span></h4>", unsafe_allow_html=True)

def image_to_base64(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

if __name__ == '__main__':
    main()
