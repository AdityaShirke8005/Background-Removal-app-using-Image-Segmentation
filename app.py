# Semantic segmentation is a deep learning algorithm that associates a label or category with every pixel in an image. 
# It is used to recognize a collection of pixels that form distinct categories.

import streamlit as st
import torch
import torchvision
from torchvision.models.segmentation import fcn_resnet101 , FCN_ResNet101_Weights

# By combining these two architectures, FCN-ResNet101 achieves state-of-the-art performance on semantic segmentation tasks, 
# allowing for pixel-level classification or labeling of objects and regions within an image.

# FCN-ResNet101 refers to a specific architecture that combines two popular deep learning models: 
# Fully Convolutional Networks (FCNs) and ResNet-101.

from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image

weights = FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1   # weights associated with coco dataset
preprocess_func = weights.transforms(resize_size=None)  # preprocessing the image - Normalization
categories = FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.meta["categories"] # Define the categories from coco dataset with weights of model (21 categories - 'aeroplane','bicycle','boat',etc)

class_to_idx = dict(zip(categories,  range(len(categories))))  # mapping categories to integer values in a dictionary

@st.cache_resource
def load_model():
    fcn_resnet = fcn_resnet101(weights=weights) 
    fcn_resnet.eval()  # pytorch syntax to make pred
    return fcn_resnet

model = load_model()

def make_prediction(processed_img):
    preds = model(processed_img.unsqueeze(dim=0))  # Input is of shape (1, 3, Width, Height)
    normalized_preds = preds["out"].squeeze().softmax(dim=0)  # Prediction is dict with keys 'out' & 'aux' . It has shape (1, 21, Width, Height)
    masks = normalized_preds > 0.5
    return masks # It will have shape (21, Width, Height)

def add_transparent_alpha_channel(pil_img):
    arr = np.array(pil_img.convert("RGBA"))  # Convert RGB to RGBA -- (W,H,3) to (W,H,4) -- 4th is alpha channel for transparency
    mask = arr[:,:,:3] == (255,255,255)  # Check for white color
    mask = mask.all(axis=2)  # Convert mask of (W,H,3) to (W,H)
    alpha = np.where(mask, 0,255)  # create aplha channel data
    arr[:,:,-1] = alpha  # Add alpha channel data
    transparent_img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")  # Convert to PIL Image
    return transparent_img

# Dashboard

from utils import set_background


set_background('bg.jpg')
st.markdown(
            "<p class='footer'>Created with ❤️ by Aditya Shirke</p>",
            unsafe_allow_html=True
        )
st.title("Background Removal app using Image Segmentation :clapper:")
st.caption("Pytorch Image Segmentation model FCN-ResNet101 to Remove Background")


upload = st.file_uploader(label="Upload image here :point_down:", type=["png","jpg","jpeg"])

if upload:
    img = Image.open(upload)
    img_tensor = torch.tensor(np.array(img).transpose(2,0,1))  # (W,H,3) --> (3,W,H)
    processed_img = preprocess_func(img_tensor)
    masks = make_prediction(processed_img) # output will be of shape (21,W,H)

    img_with_bg_removed = draw_segmentation_masks(img_tensor, masks=masks[class_to_idx["__background__"]], alpha=1.0, colors="white")
    img_with_bg_removed = to_pil_image(img_with_bg_removed)
    img_with_bg_removed = add_transparent_alpha_channel(img_with_bg_removed)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("Original Image")
        st.image(img)

    with col2:
        st.subheader("Image without Background")
        st.image(img_with_bg_removed)


