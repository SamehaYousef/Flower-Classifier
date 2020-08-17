import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
import numpy as np

def class_names(all_class_labels):
    with open(all_class_labels, 'r') as f:
        class_names = json.load(f)
        return class_names
        
        
def load_modal(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects = {'KerasLayer':hub.KerasLayer})
    print(model.summary())
    return model

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.
   
    image = image.numpy()
    return image


def predict(image_path, model_path, top_k, all_class_labels):
    
    img = Image.open(image_path)
    test_image = np.asarray(img)
    model = load_modal(model_path)
    processed_image = process_image(test_image)
    processed_image = np.expand_dims(processed_image, axis = 0)
    prob_preds = model.predict(processed_image)
    prob_preds = prob_preds[0].tolist()

    values, indices= tf.math.top_k(prob_preds, k=top_k)

    probs=values.numpy().tolist()
    classes=indices.numpy().tolist()
    
    flower_classes = class_names(all_class_labels)
    pred_label_names = [flower_classes[str(idx+1)] for idx in classes]

    return probs, pred_label_names

    