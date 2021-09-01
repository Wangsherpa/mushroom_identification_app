import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


def prepare_image(image):
    """
    This function will take an image and prepare image
    for prediction.
    :param image: (bytes) image
    :return image: tf.image (shape: [1, 224, 224, 3])
    """
    # image = tf.io.read_file(image_path)
    # image = tf.io.decode_image(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image


def predict(image, index2label, model):
    """
    This function will take image, class mapping and  trained model
    and returns predicted label and scores
    :param image: image to make prediction
    :param index2label: mapping of classname with integer label
    :param model: path to trained model
    :return
        label: predicted label (mushroom type)
        scores: predicted probabilities
    """
    image = prepare_image(image)
    prediction = model.predict(image)
    scores = prediction
    p_label = prediction.argmax(axis=-1)[0]
    return index2label[p_label], scores[0]