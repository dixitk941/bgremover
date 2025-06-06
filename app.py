# app.py

import os
import uuid
import datetime
import tempfile
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from rembg import remove
from PIL import Image, UnidentifiedImageError
import io
import logging
import shutil
import base64
from werkzeug.utils import secure_filename as werkzeug_secure_filename

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# For better temp file handling
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Use system temp directory for uploaded files in production
if os.environ.get('CLOUD_RUN', False):
    base_temp_dir = tempfile.gettempdir()
    app.config['UPLOAD_FOLDER'] = os.path.join(base_temp_dir, 'bgremover_uploads')
else:
    app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'originals'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'processed'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'jobs'), exist_ok=True)

# In-memory storage for processing status (use Redis in production)
processing_status = {}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image(file_stream):
    """Validate that the file is actually an image using PIL"""
    file_stream.seek(0)
    try:
        with Image.open(file_stream) as img:
            if img.format and img.format.lower() in [fmt.lower() for fmt in app.config['ALLOWED_EXTENSIONS']]:
                file_stream.seek(0)
                return True
            logger.warning(f"Unsupported image format: {img.format}")
    except UnidentifiedImageError:
        logger.warning("Cannot identify image file")
    except Exception as e:
        logger.warning(f"Error validating image: {str(e)}")
    
    file_stream.seek(0)
    return False

def secure_filename_with_uuid(filename):
    """Generate a secure filename with UUID to prevent collisions"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'png'
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    return f"{timestamp}_{uuid.uuid4().hex[:8]}.{ext}"

def save_job_info(job_id, job_info):
    """Save job info to file instead of session"""
    job_file = os.path.join(app.config['UPLOAD_FOLDER'], 'jobs', f"{job_id}.json")
    try:
        with open(job_file, 'w') as f:
            json.dump(job_info, f)
        processing_status[job_id] = job_info
        return True
    except Exception as e:
        logger.error(f"Error saving job info: {str(e)}")
        return False

def load_job_info(job_id):
    """Load job info from file"""
    # First check in-memory storage
    if job_id in processing_status:
        return processing_status[job_id]
    
    # Then check file storage
    job_file = os.path.join(app.config['UPLOAD_FOLDER'], 'jobs', f"{job_id}.json")
    if os.path.exists(job_file):
        try:
            with open(job_file, 'r') as f:
                job_info = json.load(f)
            processing_status[job_id] = job_info  # Cache in memory
            return job_info
        except Exception as e:
            logger.error(f"Error loading job info: {str(e)}")
    
    return None

@app.route('/')
def index():
    """Render the main landing page"""
    return render_template('index.html')

# ==================== ROUTE 1: UPLOAD ====================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """STEP 1: Handle file upload and validation ONLY"""
    if request.method == 'GET':
        # Render the upload page
        return render_template('bgremove.html')
    
    elif request.method == 'POST':
        try:
            # Check if the post request has the file part
            if 'image' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No image provided',
                    'step': 'upload'
                }), 400
            
            file = request.files['image']
            
            # Check if the file is selected
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No image selected',
                    'step': 'upload'
                }), 400
            
            # Check if the file is allowed
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'File type not allowed. Supported formats: PNG, JPG, JPEG, GIF, WEBP',
                    'step': 'upload'
                }), 400
            
            # Validate that the file is actually an image
            if not validate_image(file):
                return jsonify({
                    'success': False,
                    'error': 'Invalid image file',
                    'step': 'upload'
                }), 400
            
            # Generate unique job ID and secure filenames
            job_id = str(uuid.uuid4())
            secure_original_name = secure_filename_with_uuid(file.filename)
            original_filename = f"original_{secure_original_name}"
            
            # Save paths
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'originals', original_filename)
            
            # Save the original uploaded file
            file.save(original_path)
            
            # Optimize image size if too large (before processing)
            try:
                with Image.open(original_path) as img:
                    max_size = 1800
                    if max(img.width, img.height) > max_size:
                        ratio = max_size / max(img.width, img.height)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        try:
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                        except AttributeError:
                            img = img.resize(new_size, Image.LANCZOS)
                        img.save(original_path)
            except Exception as e:
                logger.warning(f"Error optimizing image: {str(e)}")
            
            # Store job information in file storage (NOT session)
            job_info = {
                'job_id': job_id,
                'original_filename': original_filename,
                'original_path': original_path,
                'user_filename': file.filename,
                'upload_time': datetime.datetime.now().isoformat(),
                'status': 'uploaded',
                'step': 'upload_complete'
            }
            
            # Save job info to file
            if not save_job_info(job_id, job_info):
                return jsonify({
                    'success': False,
                    'error': 'Failed to save job information',
                    'step': 'upload'
                }), 500
            
            # Create URLs for preview (don't include base64 in response)
            original_url = f"/file/originals/{original_filename}"
            
            logger.info(f"Upload completed for job {job_id}")
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'original_url': original_url,
                'filename': file.filename,
                'step': 'upload_complete',
                'message': 'Image uploaded successfully',
                'next_step': '/initiate-process',
                'redirect_url': f'/initiate-process?job_id={job_id}'
            })
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Upload failed: {str(e)}',
                'step': 'upload'
            }), 500

# ==================== ROUTE 2: INITIATE PROCESS ====================
@app.route('/initiate-process', methods=['GET', 'POST'])
def initiate_process():
    """STEP 2: Start the background removal processing"""
    if request.method == 'GET':
        # Handle GET request for URL navigation
        job_id = request.args.get('job_id')
        if not job_id:
            return redirect(url_for('upload'))
        
        # Load job info to verify it exists
        job_info = load_job_info(job_id)
        if not job_info:
            return redirect(url_for('upload'))
        
        # Return the processing page with job info
        return render_template('bgremove.html', job_id=job_id, step='initiate_process')
    
    elif request.method == 'POST':
        try:
            # Get job_id from request
            data = request.get_json() or {}
            job_id = data.get('job_id')
            
            if not job_id:
                return jsonify({
                    'success': False,
                    'error': 'No job ID provided',
                    'step': 'initiate_process'
                }), 400
            
            # Load job info from file storage
            job_info = load_job_info(job_id)
            
            if not job_info:
                return jsonify({
                    'success': False,
                    'error': 'Job not found. Please upload an image first.',
                    'step': 'initiate_process'
                }), 404
            
            original_path = job_info['original_path']
            original_filename = job_info['original_filename']
            
            # Check if file still exists
            if not os.path.exists(original_path):
                return jsonify({
                    'success': False,
                    'error': 'Original file not found. Please upload again.',
                    'step': 'initiate_process'
                }), 404
            
            # Update status to processing (setup only)
            job_info['status'] = 'processing'
            job_info['process_start_time'] = datetime.datetime.now().isoformat()
            job_info['step'] = 'processing_initiated'
            
            # Generate processed filename (but don't process yet)
            processed_filename = f"bg_removed_{original_filename.replace('original_', '')}"
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed', processed_filename)
            
            # Store processed info
            job_info['processed_filename'] = processed_filename
            job_info['processed_path'] = processed_path
            
            # Save updated job info
            save_job_info(job_id, job_info)
            
            logger.info(f"Processing initiated for job {job_id}")
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'status': 'processing_initiated',
                'step': 'processing_initiated',
                'message': 'Background removal processing initiated successfully',
                'next_step': '/background-removed',
                'redirect_url': f'/background-removed?job_id={job_id}'
            })
                
        except Exception as e:
            logger.error(f"Initiate process error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to initiate processing: {str(e)}',
                'step': 'initiate_process'
            }), 500

# ==================== ROUTE 3: BACKGROUND REMOVED ====================
@app.route('/background-removed', methods=['GET', 'POST'])
def background_removed():
    """STEP 3: Perform actual background removal and return results"""
    if request.method == 'GET':
        # Handle GET request for URL navigation
        job_id = request.args.get('job_id')
        if not job_id:
            return redirect(url_for('upload'))
        
        # Load job info to verify it exists
        job_info = load_job_info(job_id)
        if not job_info:
            return redirect(url_for('upload'))
        
        # Return the results page with job info
        return render_template('bgremove.html', job_id=job_id, step='background_removed')
    
    elif request.method == 'POST':
        try:
            # Get job status via POST
            data = request.get_json() or {}
            job_id = data.get('job_id')
            
            if not job_id:
                return jsonify({
                    'success': False,
                    'error': 'No job ID provided',
                    'step': 'background_removed'
                }), 400
            
            job_info = load_job_info(job_id)
            
            if not job_info:
                return jsonify({
                    'success': False,
                    'error': 'Job not found',
                    'step': 'background_removed'
                }), 404
            
            # If already completed, return the existing result
            if job_info['status'] == 'completed':
                logger.info(f"Job {job_id} already completed, returning existing result")
                
                try:
                    with open(job_info['original_path'], 'rb') as f:
                        original_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    with open(job_info['processed_path'], 'rb') as f:
                        processed_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Determine mime types
                    original_mime = 'image/png' if job_info['original_filename'].lower().endswith('.png') else 'image/jpeg'
                    processed_mime = 'image/png'  # rembg outputs PNG
                    
                    return jsonify({
                        'success': True,
                        'job_id': job_id,
                        'status': 'completed',
                        'step': 'background_removed',
                        'message': 'Background removal completed successfully',
                        'data': {
                            'filename': job_info['processed_filename'],
                            'original_url': job_info.get('original_url'),
                            'processed_url': job_info.get('processed_url'),
                            'original_data': f"data:{original_mime};base64,{original_base64}",
                            'processed_data': f"data:{processed_mime};base64,{processed_base64}",
                            'user_filename': job_info['user_filename']
                        },
                        'redirect_url': f'/background-removed?job_id={job_id}&completed=true'
                    })
                except Exception as e:
                    logger.error(f"Error generating base64 data for completed job {job_id}: {str(e)}")
                    return jsonify({
                        'success': False,
                        'error': 'Failed to generate download data',
                        'step': 'background_removed'
                    }), 500
            
            # If job is not in processing state, it can't be processed
            if job_info['status'] != 'processing':
                return jsonify({
                    'success': False,
                    'status': job_info['status'],
                    'step': 'background_removed',
                    'message': f"Job cannot be processed. Current status: {job_info['status']}"
                }), 400
            
            # NOW PERFORM THE ACTUAL BACKGROUND REMOVAL
            logger.info(f"Starting background removal for job {job_id}")
            
            original_path = job_info['original_path']
            processed_path = job_info['processed_path']
            
            try:
                # Process with rembg
                with open(original_path, 'rb') as input_file:
                    input_data = input_file.read()
                
                logger.info(f"Removing background for job {job_id}")
                output_data = remove(input_data)
                
                # Save the result
                with open(processed_path, 'wb') as output_file:
                    output_file.write(output_data)
                
                # Update status to completed
                job_info['status'] = 'completed'
                job_info['process_end_time'] = datetime.datetime.now().isoformat()
                job_info['step'] = 'background_removed'
                
                # Create URLs
                original_url = f"/file/originals/{job_info['original_filename']}"
                processed_url = f"/file/processed/{job_info['processed_filename']}"
                
                # Store result URLs
                job_info.update({
                    'original_url': original_url,
                    'processed_url': processed_url
                })
                
                # Save final job info
                save_job_info(job_id, job_info)
                
                logger.info(f"Background removal completed for job {job_id}")
                
                # Generate base64 data for response
                try:
                    with open(original_path, 'rb') as f:
                        original_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    with open(processed_path, 'rb') as f:
                        processed_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Determine mime types
                    original_mime = 'image/png' if job_info['original_filename'].lower().endswith('.png') else 'image/jpeg'
                    processed_mime = 'image/png'  # rembg outputs PNG
                    
                    return jsonify({
                        'success': True,
                        'job_id': job_id,
                        'status': 'completed',
                        'step': 'background_removed',
                        'message': 'Background removal completed successfully',
                        'data': {
                            'filename': job_info['processed_filename'],
                            'original_url': original_url,
                            'processed_url': processed_url,
                            'original_data': f"data:{original_mime};base64,{original_base64}",
                            'processed_data': f"data:{processed_mime};base64,{processed_base64}",
                            'user_filename': job_info['user_filename']
                        },
                        'redirect_url': f'/background-removed?job_id={job_id}&completed=true'
                    })
                except Exception as e:
                    logger.error(f"Error generating base64 data: {str(e)}")
                    return jsonify({
                        'success': False,
                        'error': 'Failed to generate download data',
                        'step': 'background_removed'
                    }), 500
                
            except Exception as e:
                # Update status to failed
                job_info['status'] = 'failed'
                job_info['error'] = str(e)
                job_info['step'] = 'processing_failed'
                save_job_info(job_id, job_info)
                
                logger.error(f"Processing error for job {job_id}: {str(e)}")
                return jsonify({
                    'success': False,
                    'status': 'failed',
                    'step': 'background_removed',
                    'error': f'Processing failed: {str(e)}'
                }), 500
                    
        except Exception as e:
            logger.error(f"Background removed error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to get result: {str(e)}',
                'step': 'background_removed'
            }), 500

# ==================== STATUS CHECK ROUTE ====================
@app.route('/check-status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Check the status of a background removal job"""
    job_info = load_job_info(job_id)
    
    if not job_info:
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404
    
    response_data = {
        'success': True,
        'job_id': job_id,
        'status': job_info['status'],
        'step': job_info.get('step', 'unknown'),
        'message': f"Job status: {job_info['status']}"
    }
    
    if job_info['status'] == 'completed':
        response_data['data'] = {
            'filename': job_info['processed_filename'],
            'original_url': job_info.get('original_url'),
            'processed_url': job_info.get('processed_url'),
            'user_filename': job_info['user_filename']
        }
    elif job_info['status'] == 'failed':
        response_data['error'] = job_info.get('error', 'Processing failed')
    
    return jsonify(response_data)

# ==================== EXISTING UTILITY ROUTES ====================
@app.route('/file/<folder>/<filename>')
def serve_file(folder, filename):
    """Serve files from the upload folder with better security"""
    if folder not in ['originals', 'processed']:
        return jsonify({'error': 'Invalid folder'}), 400
    
    safe_filename = werkzeug_secure_filename(filename)
    if safe_filename != filename:
        return jsonify({'error': 'Invalid filename'}), 400
    
    directory = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404
    
    response = send_from_directory(directory, filename)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/download/<filename>')
def download_file(filename):
    """Handle file downloads with proper headers"""
    directory = os.path.join(app.config['UPLOAD_FOLDER'], 'processed')
    response = send_from_directory(directory, filename, as_attachment=True)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/taskpage1.html')
def taskpage1():
    return render_template('taskpage1.html')

@app.route('/taskpage2.html')
def taskpage2():
    return render_template('taskpage2.html')

# ==================== ERROR HANDLERS ====================
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error. Please try again later'}), 500

# ==================== CLEANUP ====================
@app.before_request
def cleanup_old_files():
    """Clean up old files and processing status"""
    if request.endpoint == 'index' and request.method == 'GET':
        try:
            now = datetime.datetime.now()
            retention_period = datetime.timedelta(hours=1) if os.environ.get('CLOUD_RUN') else datetime.timedelta(hours=24)
            
            # Clean up files
            for folder in ['originals', 'processed', 'jobs']:
                directory = os.path.join(app.config['UPLOAD_FOLDER'], folder)
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        filepath = os.path.join(directory, filename)
                        file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                        if (now - file_modified) > retention_period:
                            os.remove(filepath)
                            logger.info(f"Removed old file: {filepath}")
            
            # Clean up processing status
            expired_jobs = []
            for job_id, job_info in processing_status.items():
                job_time = datetime.datetime.fromisoformat(job_info['upload_time'])
                if (now - job_time) > retention_period:
                    expired_jobs.append(job_id)
            
            for job_id in expired_jobs:
                del processing_status[job_id]
                logger.info(f"Removed expired job: {job_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() in ['true', '1', 't']
    
    logger.info(f"Starting application on port {port} with debug={debug}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
