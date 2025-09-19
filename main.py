from flask import Flask, request, jsonify
import os
import traceback
from utils.image_detect import classify_image_NonescapeClassifier  # adjust if needed

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})

@app.route('/classify', methods=['POST'])
def classify():
    try:
        print("Received request for image classification", request.data)
        data = request.form
        image_path = request.form.get("image_path", "").strip()

        if not image_path:
            return jsonify({'error': 'Image path is required'}), 400

        # Security: Only allow HTTPS URLs, not local file system access
        if not (image_path.startswith("https://") or image_path.startswith("http://")):
            return jsonify({'error': 'Only HTTP/HTTPS URLs are allowed for security reasons'}), 400

        result = classify_image_NonescapeClassifier(image_path)

        return jsonify({
            'result': result,
            'image_path': image_path,
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use debug=False for production deployment
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
