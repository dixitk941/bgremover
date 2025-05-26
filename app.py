# app.py

import os
import uuid
import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from rembg import remove
from PIL import Image, UnidentifiedImageError
import io
# Remove imghdr import as it's deprecated
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

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
        secure_original_name = secure_filename(file.filename)
        original_filename = f"original_{secure_original_name}"
        processed_filename = f"bg_removed_{secure_original_name}"
        
        # Save paths
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'originals', original_filename)
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed', processed_filename)
        
        # Save the original uploaded file
        file.save(original_path)
        
        # Perform background removal with error handling
        try:
            # Process with PIL to ensure compatibility
            with Image.open(original_path) as img:
                # Resize if the image is too large (optional)
                max_size = 1800
                if max(img.width, img.height) > max_size:
                    ratio = max_size / max(img.width, img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    img.save(original_path)
            
            # Process with rembg
            with open(original_path, 'rb') as input_file:
                input_data = input_file.read()
            
            # Process with rembg
            output_data = remove(input_data)
            
            # Save the result
            with open(processed_path, 'wb') as output_file:
                output_file.write(output_data)
            
            # Use URL path format (forward slashes) instead of OS path format
            # This is the key fix for your 404 error
            original_url = f"/static/uploads/originals/{original_filename}"
            processed_url = f"/static/uploads/processed/{processed_filename}"
            
            # Return JSON with the filename and URLs with forward slashes
            return jsonify({
                'success': True,
                'filename': processed_filename,
                'original_url': original_url,
                'processed_url': processed_url
            })
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

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
    """Clean up files older than 24 hours"""
    # This would be better as a scheduled task with APScheduler or similar
    # But for simplicity, we'll check occasionally on requests
    if request.endpoint == 'index' and request.method == 'GET':
        try:
            now = datetime.datetime.now()
            retention_period = datetime.timedelta(hours=24)
            
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
