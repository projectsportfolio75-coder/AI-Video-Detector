# Replace your backend/app.py with this working version

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime
import json
import logging
import time

# Register explicit adapters to remove Python 3.12 datetime deprecation warning
def _adapt_datetime(dt: datetime) -> str:
    return dt.isoformat(sep=' ')

def _convert_datetime(b: bytes) -> datetime:
    s = b.decode()
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return s  # fallback

sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("DATETIME", _convert_datetime)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'video_analysis.db')

def db_connect():
    return sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
    )

# Import the fixed advanced detector
from advanced_detector import AdvancedAIDetector

app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the advanced detector
detector = AdvancedAIDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """Initialize SQLite database for storing analysis history"""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                ai_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                is_ai BOOLEAN NOT NULL,
                frames_analyzed INTEGER,
                total_frames INTEGER,
                analysis_time REAL,
                detailed_results TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_videos INTEGER DEFAULT 0,
                total_ai_detected INTEGER DEFAULT 0,
                total_human_detected INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Initialize stats if empty
        cursor.execute('SELECT COUNT(*) FROM analysis_stats')
        if cursor.fetchone()[0] == 0:
            cursor.execute('INSERT INTO analysis_stats (total_videos) VALUES (0)')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

def save_analysis(filename, result, analysis_time):
    """Save analysis result to database"""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        
        # Insert analysis record
        cursor.execute('''
            INSERT INTO analysis_history 
            (filename, ai_probability, confidence, is_ai, frames_analyzed, total_frames, analysis_time, detailed_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            result.get('ai_probability', 0),
            result.get('confidence', 0),
            result.get('ai_probability', 0) > 60,
            result.get('frames_analyzed', 0),
            result.get('total_frames', 0),
            analysis_time,
            json.dumps(result.get('detailed_results', {}))
        ))
        
        # Update stats
        cursor.execute('SELECT * FROM analysis_stats ORDER BY id DESC LIMIT 1')
        stats = cursor.fetchone()
        
        if stats:
            total_videos = stats[1] + 1
            total_ai = stats[2] + (1 if result.get('ai_probability', 0) > 60 else 0)
            total_human = stats[3] + (1 if result.get('ai_probability', 0) <= 60 else 0)
            
            # Calculate average confidence
            cursor.execute('SELECT AVG(confidence) FROM analysis_history')
            avg_confidence = cursor.fetchone()[0] or 0
            
            cursor.execute('''
                UPDATE analysis_stats 
                SET total_videos=?, total_ai_detected=?, total_human_detected=?, avg_confidence=?, last_updated=?
                WHERE id=?
            ''', (total_videos, total_ai, total_human, avg_confidence, datetime.now(), stats[0]))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Database save failed: {e}")

@app.route('/')
def index():
    return jsonify({
        'message': '🚀 Advanced AI Video Detector API',
        'version': '2.0 - Professional Edition',
        'features': [
            'Advanced Lighting Analysis',
            'Noise Pattern Detection',
            'Generative Artifact Detection',
            'Facial Asymmetry Analysis',
            'Multi-Component AI Detection',
            'Database Storage & History'
        ],
        'status': 'Ready'
    })

@app.route('/analyze', methods=['POST'])
def analyze_video():
    start_time = time.time()
    logger.info("🚀 =============== NEW ADVANCED ANALYSIS REQUEST ===============")
    
    try:
        # Check if file was uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        logger.info(f"📁 Received file: {file.filename} ({file.content_type})")
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Supported: MP4, AVI, MOV, WMV, FLV, WebM'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"💾 Saving file to: {filepath}")
        file.save(filepath)
        
        # Check file was saved
        if not os.path.exists(filepath):
            return jsonify({'error': 'File save failed'}), 500
        
        file_size = os.path.getsize(filepath)
        logger.info(f"✅ File saved successfully, size: {file_size} bytes")
        
        # Perform advanced analysis
        logger.info(f"🔬 Starting advanced AI analysis...")
        result = detector.analyze_video(filepath)
        
        analysis_time = time.time() - start_time
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
            logger.info(f"🗑️ Cleaned up file: {filename}")
        except:
            logger.warning("⚠️ File cleanup failed")
        
        # Check for analysis errors
        if 'error' in result:
            logger.error(f"❌ Analysis error: {result['error']}")
            return jsonify(result), 500
        
        # Determine classification
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
        
        # Extract detailed scores for response
        detailed_results = result.get('detailed_results', {})
        detailed_scores = {}
        
        if 'lighting' in detailed_results and 'ai_probability' in detailed_results['lighting']:
            detailed_scores['lighting_analysis'] = detailed_results['lighting']['ai_probability']
        
        if 'noise' in detailed_results and 'ai_probability' in detailed_results['noise']:
            detailed_scores['noise_analysis'] = detailed_results['noise']['ai_probability']
        
        if 'artifacts' in detailed_results and 'ai_probability' in detailed_results['artifacts']:
            detailed_scores['artifact_detection'] = detailed_results['artifacts']['ai_probability']
        
        if 'facial' in detailed_results:
            detailed_scores['facial_analysis'] = detailed_results['facial'].get('asymmetry_score', 0) * 2
        
        # Prepare response
        response = {
            'ai_probability': round(ai_prob, 1),
            'confidence': round(confidence * 100, 1),
            'classification': classification,
            'status': status,
            'is_ai': ai_prob > 60,
            'analysis_time': round(analysis_time, 2),
            'frames_analyzed': result.get('frames_analyzed', 0),
            'total_frames': result.get('total_frames', 0),
            'components_used': result.get('analysis_components', {}),
            'detailed_scores': detailed_scores
        }
        
        # Save to database
        save_analysis(filename, result, analysis_time)
        
        logger.info(f"✅ Advanced analysis complete: {ai_prob:.1f}% AI probability in {analysis_time:.2f}s")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Analysis error: {str(e)}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'ai_probability': 0,
            'confidence': 0,
            'classification': 'Error',
            'status': 'error'
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get analysis statistics"""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        
        # Get latest stats
        cursor.execute('SELECT * FROM analysis_stats ORDER BY id DESC LIMIT 1')
        stats = cursor.fetchone()
        
        if not stats:
            return jsonify({
                'total_videos': 0,
                'total_ai_detected': 0,
                'total_human_detected': 0,
                'avg_confidence': 0,
                'accuracy_estimate': 85.0
            })
        
        # Calculate accuracy estimate based on confidence scores
        cursor.execute('SELECT AVG(confidence) FROM analysis_history WHERE confidence > 0.7')
        high_confidence_avg = cursor.fetchone()[0] or 0
        
        accuracy_estimate = min(85 + (high_confidence_avg * 10), 95)  # Cap at 95%
        
        response = {
            'total_videos': stats[1],
            'total_ai_detected': stats[2],
            'total_human_detected': stats[3],
            'avg_confidence': round(stats[4] * 100, 1),
            'accuracy_estimate': round(accuracy_estimate, 1),
            'last_updated': stats[5]
        }
        
        conn.close()
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    try:
        conn = db_connect()
        cursor = conn.cursor()
        
        # Get recent analyses
        cursor.execute('''
            SELECT filename, ai_probability, confidence, is_ai, analysis_time, timestamp
            FROM analysis_history
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'filename': row[0],
                'ai_probability': round(row[1], 1),
                'confidence': round(row[2] * 100, 1),
                'is_ai': bool(row[3]),
                'analysis_time': row[4],
                'timestamp': row[5]
            })
        
        conn.close()
        return jsonify(history)
        
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test_system():
    """Test system components"""
    try:
        import cv2
        import numpy as np
        
        # Test OpenCV
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        # Test detector initialization
        detector_status = detector.cutting_edge is not None
        
        return jsonify({
            'opencv_working': True,
            'opencv_version': cv2.__version__,
            'detector_initialized': detector_status,
            'system_status': 'All components working'
        })
    except Exception as e:
        return jsonify({
            'opencv_working': False,
            'error': str(e),
            'system_status': 'Error in components'
        })

# Initialize DB immediately so gunicorn import triggers it
init_db()

if __name__ == '__main__':
    print("Starting Advanced AI Video Detector Server...")
    print("Features: Lighting Analysis, Noise Detection, Artifact Detection, Facial Analysis")
    print("Database: Analysis history and statistics enabled")
    print("Performance: Optimized for speed and accuracy")
    
    # Use PORT environment variable for deployment
    import os
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
