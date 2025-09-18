

import argparse
import traceback
import torch
from PIL import Image
from PIL.ExifTags import TAGS
import os
from io import BytesIO
import requests
from flask import current_app
# from django.conf import settings
from nonescape import NonescapeClassifier, NonescapeClassifierMini, preprocess_image



def load_image(image_path):
    """
    Loads an image from either a local path or a URL.
    Returns a Pillow Image object in RGB mode.
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        # Load from URL into memory
        response = requests.get(image_path)
        response.raise_for_status()  # Raise error if download fails
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        # Load from local file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        return Image.open(image_path).convert("RGB")

def classify_image_NonescapeClassifier(image_path):
    try:

            # parser = argparse.ArgumentParser(description="Classify image as authentic or AI-generated")
            # parser.add_argument("model_path", help="Path to model file (.safetensors)")
            # parser.add_argument("image_path", help="Path to image file")
            # parser.add_argument("--mini", action="store_true", help="Use mini model variant")
            # args = parser.parse_args()

        # model_name = "nonescape-v0.safetensors"
        model_name = "nonescape-mini-v0.safetensors"
        # print("base dir", os.path.join('detect_ai_image', 'utils', model_name))

        model_path = "detect_ai_image/utils/nonescape-v0.safetensors"
        model_path = os.path.join('detect_ai_image', 'utils', model_name)
        # model_path = os.path.join(os.getcwd(), "detect_ai_image", "utils", "nonescape-v0.safetensors")
        model_path = os.path.join(
            current_app.root_path,  # Absolute path to your project root
            'utils', model_name
        )

        # image_path="examples/example_images/sample.jpg"


        print("Loading model...",model_path)

        # model = (NonescapeClassifier).from_pretrained(model_path)
        model = (NonescapeClassifierMini if "mini" in model_path else NonescapeClassifier).from_pretrained(model_path)

        model.eval()

        try:
            image = load_image(image_path)
            image_exif = extract_exif(image)

        except Exception as e:
            print(f"Error loading image: {e}")
            return

        tensor = preprocess_image(image)

        with torch.no_grad():
            probs = model(tensor.unsqueeze(0))
            authentic_prob = probs[0][0].item()
            ai_prob = probs[0][1].item()

        # adding by self
        verdict = "AI-generated" if ai_prob > 0.5 else "Authentic / Real"
        print(f"Verdict: {verdict}")

        confidence_level = (
        "High confidence" if abs(ai_prob - authentic_prob) > 0.4 else
        "Moderate confidence" if abs(ai_prob - authentic_prob) > 0.2 else
        "Low confidence"
        )
        print(f"Confidence Level: {confidence_level}")
        # print(f"Model: {'NonescapeClassifierMini' if args.mini else 'NonescapeClassifier'}")
        # print(f"Model Path: {args.model_path}")
        # end adding by self



        # print(f"Image: {args.image_path}")
        print(f"tensor: {probs}")
        print(f"Authentic probability: {authentic_prob:.2%}")
        print(f"Synthetic probability: {ai_prob:.2%}")

        if ai_prob > 0.5:
            print("Classification: Synthetic / AI- generated")
        else:
            print("Classification: Authentic")


        result = {
            "verdict":verdict,
            "confidence_level":confidence_level,
            "authentic_prob":authentic_prob,
            "ai_prob":ai_prob,
            "image_exif":image_exif
        }


        return result
    except Exception as e:
        print("Error during image classification:", e)
        traceback.print_exc()
        return None

def sanitize_exif_value(value):
    try:
        if isinstance(value, tuple):
            return [sanitize_exif_value(v) for v in value]
        elif hasattr(value, 'numerator') and hasattr(value, 'denominator'):
            return float(value)  # Convert IFDRational to float
        elif isinstance(value, bytes):
            return value.decode(errors='ignore')  # Decode bytes safely
        return value
    except Exception as e:
        return str(value)  # Fallback to string representation

from PIL import ExifTags
def extract_exif(image):
    try:
        print("image type check here", image)
        exif_data = image.getexif()
        exif_obj = {}
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                sanitized_value = sanitize_exif_value(value)
                print(f"{tag}: {sanitized_value}")
                exif_obj[tag] = sanitized_value

        return exif_obj
    except Exception as e:
        print("Error extracting EXIF data:", e)
        traceback.print_exc()
        return None


# if __name__ == "__main__":
#     main()
# main()
