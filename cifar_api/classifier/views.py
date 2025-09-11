# classifier/views.py
import tensorflow as tf
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from PIL import Image

# Load the trained model once
model = tf.keras.models.load_model("cifar10_cnn_model.h5")
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(image_file):
    """
    Open the uploaded image, resize to 32x32, normalize, and reshape.
    """
    img = Image.open(image_file).convert("RGB").resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)
    return img_array

@api_view(["POST"])
def predict(request):
    # Debug prints (optional)
    print("FILES received:", request.FILES)
    print("DATA received:", request.data)

    # Check if the file key exists
    if "file" not in request.FILES:
        return Response({"error": "No file uploaded"}, status=400)

    img_file = request.FILES["file"]
    img_array = preprocess_image(img_file)
    
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return Response({
        "filename": img_file.name,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4)
    })
