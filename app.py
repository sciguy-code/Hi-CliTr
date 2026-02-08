"""
Flask Web Application for X-Ray Report Generation
Uses Hi-CliTr Cognitive Radiology Model
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import io

from config import get_config
from models.model import create_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global model variable
model = None
device = None
model_trained = False
model_checkpoint = None

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():
    """Load the model on startup"""
    global model, device, model_trained, model_checkpoint
    
    # Force CPU for easier setup (GPU requires specific CUDA version)
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    print("Loading Hi-CliTr model...")
    print("   This may take a few minutes on first run...")
    
    try:
        config = get_config()
        model = create_model(config=config, pretrained=True, device=device)
        model.eval()
        print("Model architecture loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    # Try to load trained checkpoint (optional)
    model_trained = False
    model_checkpoint = None
    ckpt_env = os.environ.get("MODEL_CHECKPOINT", "").strip()
    ckpt_candidates = []
    if ckpt_env:
        ckpt_candidates.append(ckpt_env)
    else:
        ckpt_dir = Path("checkpoints")
        if ckpt_dir.exists():
            for ext in ("*.pt", "*.pth", "*.ckpt"):
                ckpt_candidates.extend([str(p) for p in ckpt_dir.glob(ext)])
            ckpt_candidates = sorted(
                ckpt_candidates,
                key=lambda p: Path(p).stat().st_mtime,
                reverse=True,
            )
    
    if ckpt_candidates:
        ckpt_path = ckpt_candidates[0]
        try:
            print(f"Loading checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict):
                state_dict = state.get("model_state_dict") or state.get("state_dict") or state
            else:
                state_dict = state
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    k = k[len("model."):]
                if k.startswith("module."):
                    k = k[len("module."):]
                cleaned[k] = v
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
            model_trained = True
            model_checkpoint = ckpt_path
            print("Checkpoint loaded; model marked as trained.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            model_trained = False
            model_checkpoint = None
    else:
        print("No checkpoint found. Model will run in untrained mode.")
    
    return model

def preprocess_image(image_bytes):

    """Preprocess uploaded image for model inference"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, image

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded X-ray image and generate report"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get clinical indication (optional)
        indication = request.form.get('indication', 'Clinical indication not provided.')
        
        # Read and preprocess image
        image_bytes = file.read()
        image_tensor, original_image = preprocess_image(image_bytes)
        
        # Generate report
        with torch.no_grad():
            result = model.generate_report(
                images=image_tensor,
                indication=indication,
                max_findings_length=150,
                max_impression_length=75,
                return_labels=model_trained
            )
        predictions = []
        if model_trained and 'predicted_labels' in result:
            labels = result['predicted_labels'][0].cpu().numpy().tolist()
            label_names = result['label_names']
            
            # Create label predictions with probabilities
            for i, (name, prob) in enumerate(zip(label_names, labels)):
                predictions.append({
                    'name': name,
                    'probability': round(prob, 3),
                    'positive': prob > 0.5
                })
            
            # Sort predictions by probability (highest first)
            predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Return response
        return jsonify({
            'success': True,
            'findings': result['findings'][0],
            'impression': result['impressions'][0],
            'report': result['reports'][0],
            'predictions': predictions,
            'model_trained': model_trained,
            'checkpoint': model_checkpoint,
            'warning': None if model_trained else 'No trained checkpoint loaded. Predictions are hidden.'
        })

        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'model_trained': model_trained,
        'checkpoint': model_checkpoint
    })

if __name__ == '__main__':
    # Load model before starting server
    load_model()
    
    print("\n" + "="*60)
    print("Hi-CliTr X-Ray Report Generator")
    print("="*60)
    print("Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
