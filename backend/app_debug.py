# Create: backend/app_debug.py
# Ultra simple version with detailed logging

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import time
import logging
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # Reduced to 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_video_ultra_simple(video_path):
    """Ultra simple analysis with detailed logging"""
    logger.info(f"🔍 STEP 1: Starting analysis of {video_path}")
    
    try:
        # Check file exists
        logger.info(f"🔍 STEP 2: Checking if file exists...")
        if not os.path.exists(video_path):
            logger.error(f"❌ File does not exist: {video_path}")
            return {'error': 'Video file not found', 'ai_probability': 0, 'confidence': 0}
        
        file_size = os.path.getsize(video_path)
        logger.info(f"🔍 STEP 3: File exists, size: {file_size} bytes")
        
        # Try to open video
        logger.info(f"🔍 STEP 4: Opening video with OpenCV...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"❌ OpenCV could not open video file")
            return {'error': 'Could not open video file', 'ai_probability': 0, 'confidence': 0}
        
        logger.info(f"✅ STEP 5: Video opened successfully")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📹 Video properties: {total_frames} frames, {fps} fps, {width}x{height}")
        
        # Read just 3 frames for ultra-fast analysis
        frames_read = 0
        frame_brightnesses = []
        
        logger.info(f"🔍 STEP 6: Reading frames...")
        
        for i in range(min(3, total_frames)):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame {i}")
                break
                
            frames_read += 1
            
            # Simple brightness calculation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            frame_brightnesses.append(brightness)
            
            logger.info(f"📊 Frame {i}: brightness = {brightness:.1f}")
        
        cap.release()
        logger.info(f"✅ STEP 7: Video processing complete, read {frames_read} frames")
        
        # Ultra simple "AI detection" based on brightness consistency
        if len(frame_brightnesses) > 1:
            brightness_std = np.std(frame_brightnesses)
            brightness_mean = np.mean(frame_brightnesses)
            
            # Very consistent brightness might indicate AI
            consistency = 1.0 - (brightness_std / (brightness_mean + 1))
            ai_probability = max(0, min(100, consistency * 80))
        else:
            ai_probability = 50  # Default if only one frame
        
        result = {
            'ai_probability': float(ai_probability),
            'confidence': 0.8,
            'frames_analyzed': frames_read,
            'total_frames': total_frames,
            'video_properties': {
                'width': width,
                'height': height,
                'fps': fps
            },
            'detailed_scores': {
                'brightness_analysis': ai_probability
            },
            'components_used': {
                'brightness_analysis': True
            }
        }
        
        logger.info(f"✅ STEP 8: Analysis complete! AI probability: {ai_probability:.1f}%")
        return result
        
    except Exception as e:
        logger.error(f"❌ STEP ERROR: Analysis failed: {str(e)}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            'error': f'Analysis failed: {str(e)}',
            'ai_probability': 0,
            'confidence': 0
        }

@app.route('/')
def index():
    return jsonify({
        'message': '🚀 Ultra Simple Debug AI Detector',
        'version': 'Debug 1.0',
        'status': 'Ready for debugging'
    })

@app.route('/analyze', methods=['POST'])
def analyze_video():
    start_time = time.time()
    logger.info("🚀 =============== NEW ANALYSIS REQUEST ===============")
    
    try:
        logger.info("📥 STEP A: Received POST request to /analyze")
        
        # Check if file was uploaded
        if 'video' not in request.files:
            logger.error("❌ No 'video' key in request.files")
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        logger.info(f"📁 STEP B: Got file object: {file}")
        logger.info(f"📁 File filename: {file.filename}")
        logger.info(f"📁 File content type: {file.content_type}")
        
        if file.filename == '':
            logger.error("❌ Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"❌ Invalid file format: {file.filename}")
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"💾 STEP C: Saving file to: {filepath}")
        
        # Save file with detailed logging
        try:
            file.save(filepath)
            logger.info(f"✅ File saved successfully")
            
            # Check saved file
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"✅ File exists on disk, size: {file_size} bytes")
            else:
                logger.error("❌ File was not saved to disk!")
                return jsonify({'error': 'File save failed'}), 500
                
        except Exception as save_error:
            logger.error(f"❌ File save error: {save_error}")
            return jsonify({'error': f'File save failed: {save_error}'}), 500
        
        # Perform analysis
        logger.info(f"🔍 STEP D: Starting video analysis...")
        result = analyze_video_ultra_simple(filepath)
        
        analysis_time = time.time() - start_time
        logger.info(f"⏱️ Total analysis time: {analysis_time:.2f} seconds")
        
        # Clean up
        try:
            os.remove(filepath)
            logger.info(f"🗑️ Cleaned up file: {filename}")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Cleanup failed: {cleanup_error}")
        
        if 'error' in result:
            logger.error(f"❌ Analysis returned error: {result['error']}")
            return jsonify(result), 500
        
        # Prepare response
        ai_prob = result.get('ai_probability', 0)
        confidence = result.get('confidence', 0)
        
        if ai_prob > 70:
            classification = 'AI-Generated'
            status = 'high_ai'
        elif ai_prob > 40:
            classification = 'Uncertain'
            status = 'uncertain'
        else:
            classification = 'Likely Human'
            status = 'likely_human'
        
        response = {
            'ai_probability': round(ai_prob, 1),
            'confidence': round(confidence * 100, 1),
            'classification': classification,
            'status': status,
            'is_ai': ai_prob > 60,
            'analysis_time': round(analysis_time, 2),
            'frames_analyzed': result.get('frames_analyzed', 0),
            'total_frames': result.get('total_frames', 0),
            'components_used': result.get('components_used', {}),
            'detailed_scores': result.get('detailed_scores', {})
        }
        
        logger.info(f"✅ STEP E: Sending response: AI={ai_prob:.1f}%, Time={analysis_time:.2f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR in analyze_video: {str(e)}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        return jsonify({
            'error': f'Server error: {str(e)}',
            'ai_probability': 0,
            'confidence': 0,
            'classification': 'Error',
            'status': 'error'
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'total_videos': 0,
        'total_ai_detected': 0,
        'total_human_detected': 0,
        'avg_confidence': 0,
        'accuracy_estimate': 85.0
    })

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify([])

@app.route('/test', methods=['GET'])
def test_opencv():
    """Test if OpenCV is working"""
    try:
        import cv2
        import numpy as np
        
        # Create a small test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        return jsonify({
            'opencv_working': True,
            'opencv_version': cv2.__version__,
            'test_brightness': float(brightness)
        })
    except Exception as e:
        return jsonify({
            'opencv_working': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("🚀 Starting Ultra Simple Debug Server...")
    print("📊 This version has detailed logging for debugging")
    print("🔧 Check the terminal for detailed logs when analyzing")
    
    app.run(debug=True, host='0.0.0.0', port=5000)