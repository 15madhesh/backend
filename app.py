import os, base64, io
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import onnxruntime as ort

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
WEIGHTS   = os.path.join(BASE_DIR, 'yolo_model', 'best.onnx')
CONF_THR  = 0.25
CLASSES   = ['Bacterial_Blight', 'Rice_Blast', 'Brown_Spot']

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

DISEASE_INFO = {
    'Bacterial_Blight': {
        'description': 'Bacterial Blight is caused by Xanthomonas oryzae pv. oryzae. It causes yellowing and wilting of leaves, leading to significant yield loss.',
        'severity': 'High',
        'treatment': [
            'Apply copper-based bactericides immediately.',
            'Remove and destroy infected plant parts.',
            'Avoid overhead irrigation to reduce leaf wetness.',
            'Use resistant rice varieties in future seasons.',
            'Ensure proper field drainage to reduce humidity.',
        ]
    },
    'Rice_Blast': {
        'description': 'Rice Blast is caused by the fungus Magnaporthe oryzae. It produces diamond-shaped lesions with grey centers on leaves and can infect all parts of the plant.',
        'severity': 'Very High',
        'treatment': [
            'Apply systemic fungicides such as Tricyclazole or Isoprothiolane.',
            'Avoid excessive nitrogen fertilization.',
            'Maintain proper spacing between plants for airflow.',
            'Remove infected debris and avoid crop residue buildup.',
            'Use certified blast-resistant rice varieties.',
        ]
    },
    'Brown_Spot': {
        'description': 'Brown Spot is caused by Helminthosporium oryzae. It appears as brown oval lesions on leaves and is often associated with nutrient-deficient soils.',
        'severity': 'Moderate',
        'treatment': [
            'Apply Mancozeb or Iprodione fungicide to affected areas.',
            'Improve soil fertility with balanced NPK fertilizers.',
            'Use disease-free certified seeds for the next planting.',
            'Avoid water stress during critical growth stages.',
            'Treat seeds with fungicides before sowing.',
        ]
    },
    'Healthy': {
        'description': 'No disease detected. The rice plant appears healthy with no visible signs of infection.',
        'severity': 'None',
        'treatment': [
            'Continue regular watering and balanced fertilization.',
            'Monitor plants weekly for any early signs of stress.',
            'Maintain good airflow between plants to prevent humidity buildup.',
            'Rotate crops annually to reduce soil-borne disease risk.',
            'Keep field clean and free from weeds.',
        ]
    }
}

print('Loading ONNX model .....')
session = None

try:
    session = ort.InferenceSession(WEIGHTS, providers=['CPUExecutionProvider'])
    print("✅ ONNX model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    session = None


def decode_b64(b64):
    if ',' in b64:
        b64 = b64.split(',')[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')


def encode_image(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def preprocess(img):
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (640, 640))
    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = img_tensor.transpose(2, 0, 1)  # HWC to CHW
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return img_tensor


def postprocess(outputs):
    pred = outputs[0][0]  # shape: (25200, 8)  → x,y,w,h, obj_conf, cls1, cls2, cls3

    # Compute final confidence = obj_conf * class_conf for each class
    obj_conf   = pred[:, 4:5]                    # (25200, 1)
    cls_scores = pred[:, 5:] * obj_conf          # (25200, 3)

    # Best class per detection
    cls_ids    = cls_scores.argmax(axis=1)       # (25200,)
    cls_confs  = cls_scores.max(axis=1)          # (25200,)

    # Filter by confidence threshold
    mask = cls_confs > CONF_THR
    cls_ids   = cls_ids[mask]
    cls_confs = cls_confs[mask]

    if len(cls_ids) == 0:
        return 'Healthy', 98, True

    best_idx   = cls_confs.argmax()
    cls_id     = int(cls_ids[best_idx])
    confidence = round(float(cls_confs[best_idx]) * 100)
    label      = CLASSES[cls_id] if cls_id < len(CLASSES) else 'Healthy'
    return label, confidence, False


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': session is not None})


@app.route('/')
def home():
    return "LeafScan Backend Running 🚀"


@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': "Missing 'image'"}), 400
            img = decode_b64(data['image'])
        elif 'image' in request.files:
            img = Image.open(request.files['image']).convert('RGB')
        else:
            return jsonify({'error': 'No image provided'}), 400

        img_tensor = preprocess(img)
        input_name = session.get_inputs()[0].name
        outputs    = session.run(None, {input_name: img_tensor})

        label, confidence, is_healthy = postprocess(outputs)
        info = DISEASE_INFO.get(label, DISEASE_INFO['Healthy'])

        return jsonify({
            'label':       label,
            'displayName': label.replace('_', ' '),
            'confidence':  confidence,
            'isHealthy':   is_healthy,
            'severity':    info['severity'],
            'description': info['description'],
            'treatment':   info['treatment'],
            'image':       encode_image(img),
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f'Model service on http://0.0.0.0:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
