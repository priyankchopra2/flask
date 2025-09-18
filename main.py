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
        # import pdb;pdb.set_trace()
        print("Received request for image classification",request.data)
        data = request.form
        image_path = request.form["image_path"]
        # user_id = data.get('user_id') or request.headers.get('X-User-ID')
        # user_email = data.get('user_email') or request.headers.get('X-User-Email')

        if not image_path:
            return jsonify({'error': 'Image path is required'}), 400

        result = classify_image_NonescapeClassifier(image_path)
        # result = ""

        return jsonify({
            'result': result,
            'image_path': image_path,
            # 'user_id': user_id,
            # 'user_email': user_email
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
