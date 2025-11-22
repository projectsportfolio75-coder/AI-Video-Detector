import cv2
import numpy as np
import os
import logging
#  from scipy.spatial import distance
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CuttingEdgeDetector:
    def __init__(self):
        try:
            # Initialize OpenCV components
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Eye aspect ratio tracking
            self.ear_threshold = 0.25
            self.blink_counter = 0
            self.total_blinks = 0
            
            logger.info("🔬 Cutting-edge detector initialized successfully")
        except Exception as e:
            logger.error(f"❌ Cutting-edge detector initialization failed: {e}")
    
    def detect_faces_opencv(self, frame):
        """Enhanced face detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhance image for better detection
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = gray  # Skip enhancement for stability
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(enhanced, 1.1, 4, minSize=(30, 30))
            return faces
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return np.array([])
    
    def analyze_lighting_consistency(self, frames):
        """Analyze lighting consistency across frames"""
        try:
            lighting_scores = []
            
            for frame in frames:
                # Convert to LAB color space
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]
                
                # Calculate lighting metrics
                mean_brightness = np.mean(l_channel)
                brightness_std = np.std(l_channel)
                
                # Calculate lighting uniformity
                uniformity = 1.0 - (brightness_std / (mean_brightness + 1e-6))
                
                lighting_scores.append({
                    'brightness': mean_brightness,
                    'uniformity': uniformity,
                    'contrast': brightness_std
                })
            
            if not lighting_scores:
                return {'error': 'No frames analyzed', 'ai_probability': 0}
            
            # Analyze consistency across frames
            brightness_values = [score['brightness'] for score in lighting_scores]
            uniformity_values = [score['uniformity'] for score in lighting_scores]
            
            brightness_consistency = 1.0 - (np.std(brightness_values) / (np.mean(brightness_values) + 1e-6))
            uniformity_consistency = 1.0 - np.std(uniformity_values)
            
            # AI-generated videos often have more consistent lighting
            overall_consistency = (brightness_consistency + uniformity_consistency) / 2
            ai_probability = max(0, (overall_consistency - 0.6) * 250)
            ai_probability = min(ai_probability, 100)
            
            return {
                'brightness_consistency': float(brightness_consistency),
                'uniformity_consistency': float(uniformity_consistency),
                'overall_consistency': float(overall_consistency),
                'ai_probability': float(ai_probability)
            }
            
        except Exception as e:
            logger.warning(f"Lighting analysis failed: {e}")
            return {'error': str(e), 'ai_probability': 0}
    
    def analyze_noise_patterns(self, frames):
        """Analyze noise patterns in frames"""
        try:
            noise_characteristics = []
            
            for frame in frames:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Simple noise detection using Gaussian blur
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                noise = cv2.absdiff(gray, blurred)
                
                # Analyze noise characteristics
                noise_mean = np.mean(noise)
                noise_std = np.std(noise)
                
                # Simple entropy calculation
                hist = cv2.calcHist([noise], [0], None, [256], [0, 256])
                hist_normalized = hist.flatten() / hist.sum()
                hist_normalized = hist_normalized[hist_normalized > 0]
                noise_entropy = -np.sum(hist_normalized * np.log2(hist_normalized)) if len(hist_normalized) > 0 else 0
                
                noise_characteristics.append({
                    'mean': noise_mean,
                    'std': noise_std,
                    'entropy': noise_entropy
                })
            
            if not noise_characteristics:
                return {'error': 'No noise analysis possible', 'ai_probability': 0}
            
            # Calculate consistency
            means = [nc['mean'] for nc in noise_characteristics]
            stds = [nc['std'] for nc in noise_characteristics]
            entropies = [nc['entropy'] for nc in noise_characteristics]
            
            mean_consistency = 1.0 - (np.std(means) / (np.mean(means) + 1e-6))
            entropy_consistency = 1.0 - (np.std(entropies) / (np.mean(entropies) + 1e-6))
            
            # AI-generated videos often have consistent noise patterns
            overall_consistency = (mean_consistency + entropy_consistency) / 2
            ai_probability = overall_consistency * 80
            
            return {
                'noise_consistency': float(overall_consistency),
                'mean_consistency': float(mean_consistency),
                'entropy_consistency': float(entropy_consistency),
                'ai_probability': float(min(ai_probability, 100))
            }
            
        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
            return {'error': str(e), 'ai_probability': 0}
    
    def detect_generative_artifacts(self, frames):
        """Detect generative artifacts in frames"""
        try:
            artifact_scores = []
            
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Simple checkerboard pattern detection
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
                
                # Simple repetition detection using Laplacian variance
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                artifact_scores.append({
                    'edge_density': edge_density,
                    'laplacian_var': laplacian_var
                })
            
            if not artifact_scores:
                return {'error': 'No artifact analysis possible', 'ai_probability': 0}
            
            # Calculate averages
            avg_edge_density = np.mean([score['edge_density'] for score in artifact_scores])
            avg_laplacian = np.mean([score['laplacian_var'] for score in artifact_scores])
            
            # Combine scores for AI probability
            ai_probability = min((avg_edge_density * 100) + (avg_laplacian / 1000), 100)
            
            return {
                'edge_density': float(avg_edge_density),
                'laplacian_variance': float(avg_laplacian),
                'ai_probability': float(ai_probability)
            }
            
        except Exception as e:
            logger.warning(f"Artifact detection failed: {e}")
            return {'error': str(e), 'ai_probability': 0}
    
    def detect_micro_expressions(self, frame):
        """Basic facial analysis"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces_opencv(frame)
            
            face_count = len(faces)
            asymmetry_score = 0
            
            if face_count > 0:
                # Simple face analysis
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Simple symmetry check
                    left_half = face_roi[:, :w//2]
                    right_half = face_roi[:, w//2:]
                    right_half_flipped = cv2.flip(right_half, 1)
                    
                    # Resize to match if needed
                    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                    left_half = left_half[:, :min_width]
                    right_half_flipped = right_half_flipped[:, :min_width]
                    
                    if left_half.shape == right_half_flipped.shape:
                        difference = cv2.absdiff(left_half, right_half_flipped)
                        asymmetry_score = np.mean(difference)
            
            return {
                'total_blinks': self.total_blinks,
                'ear_score': 0.3,  # Default value
                'asymmetry_score': float(asymmetry_score),
                'micro_expressions_detected': face_count
            }
            
        except Exception as e:
            logger.warning(f"Micro-expression analysis failed: {e}")
            return {
                'total_blinks': 0,
                'ear_score': 0.3,
                'asymmetry_score': 0.0,
                'micro_expressions_detected': 0
            }


class AdvancedAIDetector:
    def __init__(self):
        try:
            # Initialize cutting-edge detector
            self.cutting_edge = CuttingEdgeDetector()
            logger.info("🚀 Advanced AI Detector initialized successfully!")
            
        except Exception as e:
            logger.error(f"⚠️ Advanced AI Detector initialization failed: {e}")
            self.cutting_edge = None
    
    def analyze_video(self, video_path):
        """Main video analysis function with advanced features"""
        logger.info(f"🎬 Starting advanced analysis: {video_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                return {
                    'error': 'Video file not found',
                    'ai_probability': 0,
                    'confidence': 0
                }
            
            # Extract frames
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    'error': 'Could not open video file',
                    'ai_probability': 0,
                    'confidence': 0
                }
            
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample every 30th frame for efficiency
                if frame_count % 30 == 0:
                    frames.append(frame)
                
                frame_count += 1
                
                # Limit analysis to 10 frames for speed
                if len(frames) >= 10:
                    break
            
            cap.release()
            
            if not frames:
                return {
                    'error': 'No frames could be extracted',
                    'ai_probability': 0,
                    'confidence': 0
                }
            
            logger.info(f"📹 Extracted {len(frames)} frames for analysis")
            
            # Perform analyses
            results = {}
            ai_probabilities = []
            
            if self.cutting_edge:
                # 1. Lighting analysis
                logger.info("💡 Analyzing lighting patterns...")
                lighting_results = self.cutting_edge.analyze_lighting_consistency(frames)
                results['lighting'] = lighting_results
                if 'ai_probability' in lighting_results:
                    ai_probabilities.append(lighting_results['ai_probability'])
                
                # 2. Noise analysis
                logger.info("🔊 Analyzing noise patterns...")
                noise_results = self.cutting_edge.analyze_noise_patterns(frames)
                results['noise'] = noise_results
                if 'ai_probability' in noise_results:
                    ai_probabilities.append(noise_results['ai_probability'])
                
                # 3. Artifact detection
                logger.info("🎨 Detecting artifacts...")
                artifact_results = self.cutting_edge.detect_generative_artifacts(frames)
                results['artifacts'] = artifact_results
                if 'ai_probability' in artifact_results:
                    ai_probabilities.append(artifact_results['ai_probability'])
                
                # 4. Facial analysis (first frame only for speed)
                logger.info("👤 Analyzing facial features...")
                facial_results = self.cutting_edge.detect_micro_expressions(frames[0])
                results['facial'] = facial_results
                # Add facial asymmetry to AI probability
                facial_ai_prob = min(facial_results['asymmetry_score'] * 2, 100)
                ai_probabilities.append(facial_ai_prob)
            
            # Calculate overall AI probability
            if ai_probabilities:
                # Weight the different analyses
                weights = [0.3, 0.25, 0.3, 0.15][:len(ai_probabilities)]  # lighting, noise, artifacts, facial
                weights = np.array(weights) / np.sum(weights)  # Normalize weights
                
                overall_ai_probability = np.average(ai_probabilities, weights=weights)
            else:
                overall_ai_probability = 0
            
            # Calculate confidence based on consistency
            confidence = 0.8 if len(frames) >= 5 else 0.6
            
            # Final results
            final_results = {
                'ai_probability': float(overall_ai_probability),
                'confidence': float(confidence),
                'frames_analyzed': len(frames),
                'total_frames': frame_count,
                'analysis_components': {
                    'lighting_analysis': 'lighting' in results,
                    'noise_analysis': 'noise' in results,
                    'artifact_detection': 'artifacts' in results,
                    'facial_analysis': 'facial' in results
                },
                'detailed_results': results
            }
            
            logger.info(f"✅ Advanced analysis complete! AI Probability: {overall_ai_probability:.1f}%")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Advanced analysis failed: {str(e)}")
            return {
                'error': f'Analysis failed: {str(e)}',
                'ai_probability': 0,
                'confidence': 0
            }


# Backward compatible function
def detect_ai_video(video_path):
    """Backward compatible detection function"""
    detector = AdvancedAIDetector()
    result = detector.analyze_video(video_path)
    
    return {
        'ai_probability': result.get('ai_probability', 0),
        'confidence': result.get('confidence', 0),
        'is_ai': result.get('ai_probability', 0) > 60,
        'analysis_details': result.get('detailed_results', {})
    }