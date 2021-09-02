import pickle
import streamlit as st
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from predict import predict

# Display app header
st.header("Mushroom Identification App")


@st.cache()
def load_files():
    with open('data/mushroom_index2label.pickle', 'rb') as handle:
        index2label = pickle.load(handle)

    model = load_model("data/mushroom_model.h5")

    return model, index2label


# load model and classmap files
model, index2label = load_files()

# file uploader to upload image
image_uploaded = st.file_uploader("Upload mushroom image")
col1, col2 = st.beta_columns((1.2, 2))
if image_uploaded:

    image = tf.io.decode_image(image_uploaded.read())
    image_copy = tf.image.resize(image, [300, 300])
    with col1:
        st.image(image_copy.numpy().squeeze().astype(np.uint8), clamp=True, channels='RGB')

    if st.button("Predict"):

        # predict mushroom type
        label, scores = predict(image, index2label, model)

        # display prediction
        st.success("This mushroom looks like {}\n".format(label.capitalize()))

        # plot probabilities
        class_score = [(index2label[i], score) for i, score in enumerate(scores)]
        class_names = [c[0] for c in class_score]
        class_scores = [c[1] for c in class_score]

        # set font size
        sns.set(font_scale=0.5)
        # get a matplotlib figure
        fig = plt.figure(figsize=(4, 2))
        # plot a barplot
        ax = sns.barplot(x=class_names, y=class_scores, errwidth=5)
        # show percentages on top
        patches = ax.patches
        for i in range(len(patches)):
            x = patches[i].get_x() + patches[i].get_width() / 2
            y = patches[i].get_height() + .05
            ax.annotate('{:.1f}%'.format(scores[i] * 100), (x, y), ha='center')
        plt.title("Confidence of a Model", fontsize=10)
        plt.ylabel("Probability")
        plt.xlabel("Mushroom Type")
        # display figure
        with col2:
            st.pyplot(fig)
