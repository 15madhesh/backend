import os, sys, base64, io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
WEIGHTS   = os.path.join(BASE_DIR, 'yolo_model', 'best.pt')
CONF_THR  = 0.25

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

print('Loading YOLO model .....')
model = None

try:
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    model.conf = CONF_THR
    print("✅ Model loaded successfully (ultralytics)")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None


def decode_b64(b64):
    if ',' in b64:
        b64 = b64.split(',')[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')


def encode_image(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
    })


@app.route('/')
def home():
    return "LeafScan Backend Running 🚀"


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # -------- IMAGE INPUT --------
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': "Missing 'image'"}), 400
            img = decode_b64(data['image'])
        elif 'image' in request.files:
            img = Image.open(request.files['image']).convert('RGB')
        else:
            return jsonify({'error': 'No image provided'}), 400

        # -------- PREDICTION (ultralytics API) --------
        results = model(img, imgsz=640)
        result  = results[0]
        boxes   = result.boxes

        if boxes is None or len(boxes) == 0:
            label      = 'Healthy'
            confidence = 98
            is_healthy = True
        else:
            # Pick detection with highest confidence
            confs = boxes.conf.cpu().numpy()
            best_idx   = int(confs.argmax())
            cls_id     = int(boxes.cls[best_idx].cpu().numpy())
            label      = result.names[cls_id]
            confidence = round(float(confs[best_idx]) * 100)
            is_healthy = False

        # -------- DRAW OUTPUT IMAGE --------
        plotted    = result.plot()          # numpy BGR array
        output_img = Image.fromarray(plotted[..., ::-1])  # BGR→RGB
        img_base64 = encode_image(output_img)

        # -------- INFO --------
        info = DISEASE_INFO.get(label, DISEASE_INFO['Healthy'])

        return jsonify({
            'label':       label,
            'displayName': label.replace('_', ' '),
            'confidence':  confidence,
            'isHealthy':   is_healthy,
            'severity':    info['severity'],
            'description': info['description'],
            'treatment':   info['treatment'],
            'image':       img_base64,
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f'Model service on http://0.0.0.0:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
