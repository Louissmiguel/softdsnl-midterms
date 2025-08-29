from rest_framework.decorators import api_view
from rest_framework.response import Response
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once (faster than loading every request)
model = tf.keras.models.load_model("cifar10_cnn_model.h5")

# CIFAR-10 class labels
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@api_view(["POST"])
def predict_image(request):
    try:
        # 1. Get the uploaded image
        file = request.FILES["file"]
        img = Image.open(file).resize((32, 32))  # CIFAR-10 size
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)  # shape (1, 32, 32, 3)

        # 2. Predict
        predictions = model.predict(img)
        predicted_class = class_names[np.argmax(predictions)]

        return Response({
            "predicted_class": predicted_class,
            "confidence": float(np.max(predictions))
        })

    except Exception as e:
        return Response({"error": str(e)})
