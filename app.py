# app.py

import os
import uuid
import datetime
import tempfile
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from rembg import remove
from PIL import Image, UnidentifiedImageError
import io
import logging
import shutil
from werkzeug.utils import secure_filename as werkzeug_secure_filename

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# For better temp file handling
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Use system temp directory for uploaded files in production
if os.environ.get('CLOUD_RUN', False):
    # Create a subfolder in the system temp directory
    base_temp_dir = tempfile.gettempdir()
    app.config['UPLOAD_FOLDER'] = os.path.join(base_temp_dir, 'bgremover_uploads')
else:
    # Use static/uploads for local development
    app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'originals'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'processed'), exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image(file_stream):
    """Validate that the file is actually an image using PIL instead of imghdr"""
    file_stream.seek(0)
    try:
        # Try to open the image with PIL
        with Image.open(file_stream) as img:
            # Check if the format is supported
            if img.format and img.format.lower() in [fmt.lower() for fmt in app.config['ALLOWED_EXTENSIONS']]:
                file_stream.seek(0)  # Reset file pointer
                return True
            logger.warning(f"Unsupported image format: {img.format}")
    except UnidentifiedImageError:
        logger.warning("Cannot identify image file")
    except Exception as e:
        logger.warning(f"Error validating image: {str(e)}")
    
    file_stream.seek(0)  # Reset file pointer even on error
    return False

def secure_filename(filename):
    """Generate a secure filename with UUID to prevent collisions"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'png'
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    return f"{timestamp}_{uuid.uuid4().hex[:8]}.{ext}"

@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and background removal"""
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Check if the file is selected
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Check if the file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Supported formats: PNG, JPG, JPEG, GIF, WEBP'}), 400
        
        # Validate that the file is actually an image
        if not validate_image(file):
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Create secure filenames
        original_filename = f"original_{secure_filename(file.filename)}"
        processed_filename = f"bg_removed_{secure_filename(file.filename)}"
        
        # Save paths
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'originals', original_filename)
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed', processed_filename)
        
        # Save the original uploaded file
        file.save(original_path)
        
        # Perform background removal with error handling
        try:
            # Process with PIL to ensure compatibility and optimize
            with Image.open(original_path) as img:
                # Resize if the image is too large to conserve memory
                max_size = 1800
                if max(img.width, img.height) > max_size:
                    ratio = max_size / max(img.width, img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    try:
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    except AttributeError:
                        # Fall back for older Pillow versions
                        img = img.resize(new_size, Image.LANCZOS)
                    img.save(original_path)
            
            # Process with rembg - use a separate try/except for better error handling
            try:
                with open(original_path, 'rb') as input_file:
                    input_data = input_file.read()
                
                logger.info(f"Removing background from {original_filename}")
                output_data = remove(input_data)
                
                # Save the result
                with open(processed_path, 'wb') as output_file:
                    output_file.write(output_data)
            except Exception as rembg_error:
                logger.error(f"Error in rembg processing: {str(rembg_error)}")
                raise
            
            # Create URLs to access the images
            # Use relative URLs that work in both environments
            original_url = f"/file/originals/{original_filename}"
            processed_url = f"/file/processed/{processed_filename}"
            
            logger.info(f"Successfully processed image: {processed_filename}")
            
            # Return JSON with proper URLs
            return jsonify({
                'success': True,
                'filename': processed_filename,
                'original_url': original_url,
                'processed_url': processed_url
            })
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            # Clean up any partial files
            for path in [original_path, processed_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/file/<folder>/<filename>')
def serve_file(folder, filename):
    """Serve files from the upload folder with better security"""
    # Validate folder
    if folder not in ['originals', 'processed']:
        return jsonify({'error': 'Invalid folder'}), 400
    
    # Validate filename to prevent directory traversal
    safe_filename = werkzeug_secure_filename(filename)
    if safe_filename != filename:
        return jsonify({'error': 'Invalid filename'}), 400
    
    directory = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    
    # Verify the file exists
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Serve the file
    try:
        return send_from_directory(directory, filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': 'Error serving file'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Handle file downloads with proper headers"""
    directory = os.path.join(app.config['UPLOAD_FOLDER'], 'processed')
    response = send_from_directory(directory, filename, as_attachment=True)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size too large errors"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Server error. Please try again later'}), 500

# Clean up old files periodically (optional)
@app.before_request
def cleanup_old_files():
    """Clean up files older than 1 hour in production (Cloud Run has ephemeral storage)"""
    if request.endpoint == 'index' and request.method == 'GET':
        try:
            now = datetime.datetime.now()
            # Shorter retention period for Cloud Run
            retention_period = datetime.timedelta(hours=1) if os.environ.get('CLOUD_RUN') else datetime.timedelta(hours=24)
            
            for folder in ['originals', 'processed']:
                directory = os.path.join(app.config['UPLOAD_FOLDER'], folder)
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                    if now - file_modified > retention_period:
                        os.remove(filepath)
                        logger.info(f"Removed old file: {filepath}")
        except Exception as e:
            logger.error(f"Error cleaning up files: {str(e)}")

# Health check endpoint for Cloud Run
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Set up proper startup based on environment
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() in ['true', '1', 't']
    
    # Log application startup
    logger.info(f"Starting application on port {port} with debug={debug}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    # Run the application
    app.run(debug=debug, host='0.0.0.0', port=port)
