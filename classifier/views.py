import numpy as np
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model

# Load CIFAR-10 model once
model = load_model("cifar10_cnn_model.h5")
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@csrf_exempt
def predict(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
            data = np.array(body["data"])  # Expect 32x32x3 image array

            # Ensure correct shape
            if data.shape != (32, 32, 3):
                return JsonResponse({"error": "Input must be 32x32x3"}, status=400)

            # Preprocess (normalize)
            data = data.astype("float32") / 255.0
            data = np.expand_dims(data, axis=0)  # add batch dimension → (1,32,32,3)

            preds = model.predict(data)
            predicted_class = class_names[np.argmax(preds[0])]
            confidence = float(np.max(preds[0]))

            return JsonResponse({
                "prediction": predicted_class,
                "confidence": round(confidence, 4)
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "POST request required"}, status=400)
