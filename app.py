# app.py

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, make_response, abort
from werkzeug.utils import secure_filename as werkzeug_secure_filename
import os
import uuid
import json
from datetime import datetime, timedelta
import logging
from rembg import remove, new_session
from PIL import Image, UnidentifiedImageError, ImageFilter, ImageEnhance, ImageOps
import io
import shutil
import base64
import tempfile
import numpy as np
import threading
import time
import atexit
from scipy import ndimage
import cv2
import gc
import psutil

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

# ==================== MEMORY-OPTIMIZED MODEL LOADING SYSTEM ====================
# Uses ONLY ONE model at a time to minimize RAM usage
# Critical for servers with limited RAM (8GB + swap)

# Memory monitoring helper
def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def force_garbage_collection():
    """Force garbage collection to free memory"""
    gc.collect()
    # Also clear any PyTorch/ONNX caches if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()

class MemoryOptimizedModelLoader:
    """Memory-optimized model loader - keeps ONLY ONE model loaded at a time"""
    
    def __init__(self):
        self._current_model = None
        self._current_model_key = None
        self._model_names = {
            'isnet': 'isnet-general-use',  # Lightweight, good quality (~170MB)
            'u2net': 'u2net',               # Medium size (~170MB)
            'birefnet': 'birefnet-general', # High quality but larger (~350MB)
            'human': 'u2net_human_seg',     # For humans (~170MB)
        }
        # Default model - isnet is lightweight and good quality
        self._default_model = 'isnet'
        self._lock = threading.Lock()
        logger.info("Memory-optimized model loader initialized (single model mode)")
    
    def get_model(self, model_key=None):
        """Get a model, unloading any previous model first to save memory"""
        if model_key is None:
            model_key = self._default_model
        
        # Map old model keys to optimized ones
        key_mapping = {
            'birefnet_hq': 'birefnet',  # Don't use massive model
            'general': 'u2net',
            'fine': 'isnet',
        }
        model_key = key_mapping.get(model_key, model_key)
        
        with self._lock:
            # If same model is already loaded, return it
            if self._current_model_key == model_key and self._current_model is not None:
                return self._current_model
            
            # Unload current model first to free memory
            if self._current_model is not None:
                mem_before = get_memory_usage_mb()
                logger.info(f"Unloading model '{self._current_model_key}' to free memory (current: {mem_before:.0f}MB)")
                self._current_model = None
                self._current_model_key = None
                force_garbage_collection()
                mem_after = get_memory_usage_mb()
                logger.info(f"Memory after unload: {mem_after:.0f}MB (freed: {mem_before - mem_after:.0f}MB)")
            
            # Load the new model
            model_name = self._model_names.get(model_key, self._model_names[self._default_model])
            logger.info(f"Loading model: {model_name} (memory: {get_memory_usage_mb():.0f}MB)")
            
            try:
                self._current_model = new_session(model_name)
                self._current_model_key = model_key
                logger.info(f"✓ Loaded {model_name} (memory: {get_memory_usage_mb():.0f}MB)")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                # Fallback to lightweight isnet
                if model_key != 'isnet':
                    logger.info("Falling back to isnet model")
                    return self.get_model('isnet')
                raise e
            
            return self._current_model
    
    def unload_all(self):
        """Unload all models to free memory"""
        with self._lock:
            if self._current_model is not None:
                logger.info(f"Unloading all models (memory before: {get_memory_usage_mb():.0f}MB)")
                self._current_model = None
                self._current_model_key = None
                force_garbage_collection()
                logger.info(f"All models unloaded (memory after: {get_memory_usage_mb():.0f}MB)")
    
    def get_loaded_models(self):
        """Return list of currently loaded models"""
        if self._current_model_key:
            return [self._current_model_key]
        return []

# Initialize memory-optimized model loader (NO preloading - saves ~500MB+ RAM on startup)
model_loader = MemoryOptimizedModelLoader()
logger.info("✓ Model loader initialized - models will load on first request (saves RAM)")

# Compatibility aliases for existing code
def get_rembg_session_birefnet():
    return model_loader.get_model('birefnet')

def get_rembg_session_birefnet_hq():
    return model_loader.get_model('birefnet_hq')

def get_rembg_session_general():
    return model_loader.get_model('general')

def get_rembg_session_fine():
    return model_loader.get_model('fine')

def get_rembg_session_human():
    return model_loader.get_model('human')

# For backward compatibility - these are now functions, not variables
# The code that uses these will need to call them as functions
rembg_session_birefnet = None  # Use get_rembg_session_birefnet() instead
rembg_session_birefnet_hq = None  # Use get_rembg_session_birefnet_hq() instead
rembg_session_general = None  # Use get_rembg_session_general() instead
rembg_session_fine = None  # Use get_rembg_session_fine() instead
rembg_session_human = None  # Use get_rembg_session_human() instead

# ==================== ENHANCED EDGE DETECTION FUNCTIONS ====================

def analyze_image_content(image_np):
    """Analyze image to determine best processing approach"""
    # Convert to different color spaces for analysis
    if len(image_np.shape) == 2:
        gray = image_np
    else:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Detect skin tones (indicates human subject)
    if len(image_np.shape) == 3:
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        # Skin color range in HSV - more restrictive
        lower_skin1 = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin1 = np.array([20, 150, 255], dtype=np.uint8)
        lower_skin2 = np.array([0, 10, 60], dtype=np.uint8)
        upper_skin2 = np.array([25, 255, 255], dtype=np.uint8)
        skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        skin_percentage = np.sum(skin_mask > 0) / skin_mask.size
        has_human = skin_percentage > 0.08  # More than 8% skin tones
    else:
        has_human = False
    
    # Detect complexity (edges density)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    is_complex = edge_density > 0.05
    
    # Detect sharp/hard edges (indicates product/object vs organic subject)
    # Products have more straight lines and sharp corners
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    has_straight_lines = lines is not None and len(lines) > 5
    
    # Detect if object has defined boundaries (high contrast at edges)
    # Use Laplacian variance to measure sharpness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    has_sharp_edges = laplacian_var > 500
    
    # Detect if background is relatively uniform
    h, w = gray.shape[:2]
    border_size = min(20, h // 10, w // 10)
    border_pixels = np.concatenate([
        gray[:border_size, :].flatten(),
        gray[-border_size:, :].flatten(),
        gray[:, :border_size].flatten(),
        gray[:, -border_size:].flatten()
    ])
    bg_uniformity = np.std(border_pixels)
    has_uniform_bg = bg_uniformity < 40
    
    # ===== DETECT LOGO/GRAPHIC =====
    # Logos typically have:
    # - Few unique colors (flat colors)
    # - Very uniform regions
    # - High contrast between regions
    # - Geometric shapes
    
    # Count unique colors (simplified by quantizing)
    if len(image_np.shape) == 3:
        # Quantize to reduce color count
        quantized = (image_np // 32) * 32  # Reduce to ~8 levels per channel
        unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        is_logo = unique_colors < 50  # Logos have very few colors
        
        # Also check for large uniform regions (typical of logos/icons)
        # Count pixels that are very close to pure colors
        pure_black = np.all(image_np < 30, axis=2)
        pure_white = np.all(image_np > 225, axis=2)
        saturated = hsv[:, :, 1] > 200  # Highly saturated colors
        
        uniform_ratio = (np.sum(pure_black) + np.sum(pure_white) + np.sum(saturated)) / gray.size
        has_flat_colors = uniform_ratio > 0.3  # 30% of image is flat colors
        
        is_logo = is_logo or (has_flat_colors and unique_colors < 100)
    else:
        is_logo = False
        unique_colors = 256
    
    # ===== DETECT DOCUMENT/ID CARD =====
    # Documents typically have:
    # - Rectangular shape filling most of the image
    # - Text (detected by many small connected components)
    # - Often have a face/photo in them
    # - Aspect ratio close to standard card sizes (85.6mm x 53.98mm = 1.586 ratio)
    
    h, w = gray.shape[:2]
    aspect_ratio = max(w, h) / min(w, h)
    
    # Check if aspect ratio matches common document sizes
    # Credit card/ID: 1.586, A4: 1.414, Letter: 1.294
    is_card_ratio = 1.4 < aspect_ratio < 1.8
    
    # Detect text-like features (many small high-contrast regions)
    # Use adaptive threshold to find text
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
    # Count connected components (text has many small components)
    num_labels, labels = cv2.connectedComponents(adaptive_thresh)
    has_text_like_features = num_labels > 100  # Documents have many small regions
    
    # Check if image has a clear rectangular boundary (document edge)
    # Look for strong edges near the border
    edge_border = 30
    border_edges = np.concatenate([
        edges[:edge_border, :].flatten(),
        edges[-edge_border:, :].flatten(),
        edges[:, :edge_border].flatten(),
        edges[:, -edge_border:].flatten()
    ])
    has_border_edges = np.sum(border_edges > 0) / border_edges.size > 0.05
    
    # Detect if there's a photo/face region (common in ID cards)
    # Look for a rectangular region with different characteristics
    has_photo_region = False
    if len(image_np.shape) == 3:
        # Check for face-like region (small rectangular area with skin tones)
        # ID card photos are usually in a corner
        quarter_h, quarter_w = h // 4, w // 4
        corners = [
            image_np[:quarter_h, :quarter_w],  # Top-left
            image_np[:quarter_h, -quarter_w:],  # Top-right
            image_np[-quarter_h:, :quarter_w],  # Bottom-left
            image_np[-quarter_h:, -quarter_w:]  # Bottom-right
        ]
        for corner in corners:
            corner_hsv = cv2.cvtColor(corner, cv2.COLOR_RGB2HSV)
            corner_skin = cv2.inRange(corner_hsv, lower_skin1, upper_skin1)
            if np.sum(corner_skin > 0) / corner_skin.size > 0.15:  # 15% skin in corner
                has_photo_region = True
                break
    
    # Document detection: has text + card ratio OR has text + photo + edges
    is_document = (has_text_like_features and is_card_ratio) or \
                  (has_text_like_features and has_photo_region and has_border_edges) or \
                  (has_text_like_features and has_border_edges and is_card_ratio)
    
    # Determine if likely a product/object (not human)
    is_product = has_straight_lines or (has_sharp_edges and not has_human)
    
    # Documents should not be treated as products
    if is_document:
        is_product = False
    
    return {
        'has_human': has_human,
        'is_product': is_product,
        'is_logo': is_logo,
        'is_document': is_document,
        'is_complex': is_complex,
        'has_uniform_bg': has_uniform_bg,
        'has_sharp_edges': has_sharp_edges,
        'edge_density': edge_density,
        'bg_uniformity': bg_uniformity,
        'laplacian_var': laplacian_var
    }

def apply_grabcut_refinement(image_rgba, original_rgb):
    """Apply GrabCut algorithm for precise edge refinement on objects"""
    try:
        # Get current alpha mask
        r, g, b, a = image_rgba.split()
        alpha_np = np.array(a, dtype=np.uint8)
        original_np = np.array(original_rgb)
        
        # Create initial mask from alpha
        mask = np.zeros(alpha_np.shape, np.uint8)
        mask[alpha_np > 200] = cv2.GC_FGD  # Definite foreground
        mask[alpha_np < 30] = cv2.GC_BGD   # Definite background
        mask[(alpha_np >= 30) & (alpha_np <= 200)] = cv2.GC_PR_FGD  # Probable foreground
        
        # Apply GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Convert to BGR for OpenCV
        bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        
        cv2.grabCut(bgr, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        
        # Create refined mask
        refined_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        
        # Blend with original alpha (GrabCut can sometimes be too aggressive)
        alpha_refined = (alpha_np.astype(np.float32) * 0.6 + refined_mask.astype(np.float32) * 0.4).astype(np.uint8)
        
        alpha_pil = Image.fromarray(alpha_refined)
        return Image.merge('RGBA', (r, g, b, alpha_pil))
    except Exception as e:
        logger.warning(f"GrabCut refinement failed: {e}")
        return image_rgba

def detect_multiple_objects(alpha_mask):
    """Detect if there are multiple separate objects in the image"""
    # Convert to binary mask
    binary_mask = (alpha_mask > 128).astype(np.uint8)
    
    # Label connected components
    labeled_array, num_features = ndimage.label(binary_mask)
    
    # Get sizes of each component
    component_sizes = ndimage.sum(binary_mask, labeled_array, range(1, num_features + 1))
    
    # Filter out tiny components (noise) - keep only components > 1% of largest
    if len(component_sizes) > 0:
        max_size = max(component_sizes)
        significant_components = sum(1 for size in component_sizes if size > max_size * 0.01)
        return significant_components, labeled_array, component_sizes
    
    return 0, labeled_array, []

def refine_edges_advanced(image_rgba, edge_refinement_strength=1.0, is_product=False):
    """Advanced edge refinement using morphological operations and edge-aware smoothing"""
    r, g, b, a = image_rgba.split()
    alpha_np = np.array(a, dtype=np.float32)
    rgb_np = np.array(image_rgba.convert('RGB'))
    
    # Convert to uint8 for OpenCV operations
    alpha_uint8 = alpha_np.astype(np.uint8)
    
    # Step 1: Detect edges in the alpha channel
    edges = cv2.Canny(alpha_uint8, 30, 100)
    
    # Step 2: Also detect edges in the RGB image for reference
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(gray, 50, 150)
    
    # Combine edge information
    combined_edges = cv2.bitwise_or(edges, rgb_edges)
    
    # Step 3: Create edge mask with dilation for the transition zone
    # Use smaller kernels for products to preserve sharp edges
    if is_product:
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Rect for sharp edges
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        edge_zone = cv2.dilate(combined_edges, kernel_tiny, iterations=1)  # Less dilation
    else:
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_zone = cv2.dilate(combined_edges, kernel_small, iterations=2)
    
    # Step 4: Clean up the alpha channel - remove halos
    if is_product:
        # For products, be less aggressive with erosion to preserve edges
        alpha_cleaned = alpha_uint8.copy()
        # Only light erosion
        alpha_cleaned = cv2.erode(alpha_cleaned, kernel_tiny, iterations=1)
        alpha_cleaned = cv2.dilate(alpha_cleaned, kernel_tiny, iterations=1)
    else:
        # For humans, standard erosion/dilation
        alpha_cleaned = cv2.erode(alpha_uint8, kernel_tiny, iterations=1)
        alpha_cleaned = cv2.dilate(alpha_cleaned, kernel_tiny, iterations=1)
    
    # Remove small holes
    alpha_cleaned = cv2.morphologyEx(alpha_cleaned, cv2.MORPH_CLOSE, kernel_small)
    # Remove small noise
    alpha_cleaned = cv2.morphologyEx(alpha_cleaned, cv2.MORPH_OPEN, kernel_tiny)
    
    # Step 5: Create a refined mask using grabcut-like approach on edge zones
    # Different thresholds for products vs humans
    edge_mask = edge_zone > 0
    if is_product:
        # Higher threshold for products - preserve more detail
        alpha_cleaned[edge_mask & (alpha_cleaned < 128)] = 0
        alpha_cleaned[edge_mask & (alpha_cleaned >= 128)] = 255
    else:
        alpha_cleaned[edge_mask & (alpha_cleaned < 180)] = 0
        alpha_cleaned[edge_mask & (alpha_cleaned >= 180)] = 255
    
    # Step 6: Apply guided filter for edge-aware smoothing (if available) or bilateral
    if is_product:
        # Less smoothing for products to keep sharp edges
        alpha_smoothed = cv2.bilateralFilter(alpha_cleaned, 3, 30, 30)
    else:
        alpha_smoothed = cv2.bilateralFilter(alpha_cleaned, 5, 50, 50)
    
    # Step 7: Blend original and smoothed based on edge zone
    edge_zone_float = (edge_zone.astype(np.float32) / 255.0) * edge_refinement_strength
    if is_product:
        # Less blending for products
        alpha_result = alpha_cleaned.astype(np.float32) * (1 - edge_zone_float * 0.5) + \
                       alpha_smoothed.astype(np.float32) * edge_zone_float * 0.5
    else:
        alpha_result = alpha_cleaned.astype(np.float32) * (1 - edge_zone_float) + \
                       alpha_smoothed.astype(np.float32) * edge_zone_float
    
    # Step 8: Apply strong threshold to remove semi-transparency artifacts (halos)
    if is_product:
        # More aggressive threshold for products - no semi-transparency
        alpha_result = np.where(alpha_result < 80, 0, alpha_result)
        alpha_result = np.where(alpha_result > 175, 255, alpha_result)
    else:
        alpha_result = np.where(alpha_result < 40, 0, alpha_result)
        alpha_result = np.where(alpha_result > 215, 255, alpha_result)
    
    # Step 9: Very subtle gaussian blur for natural transition at remaining semi-transparent areas
    alpha_pil = Image.fromarray(alpha_result.astype(np.uint8))
    if is_product:
        # No blur for products - keep sharp edges
        pass
    else:
        alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    return Image.merge('RGBA', (r, g, b, alpha_pil))

def enhance_edge_detection(image_rgba, original_image):
    """Use edge detection from original image to improve mask boundaries"""
    # Convert original to grayscale for edge detection
    if original_image.mode != 'RGB':
        original_rgb = original_image.convert('RGB')
    else:
        original_rgb = original_image
    
    original_np = np.array(original_rgb)
    gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
    
    # Detect edges using Canny
    edges = cv2.Canny(gray, 100, 200)
    
    # Dilate edges slightly to create edge zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_zone = cv2.dilate(edges, kernel, iterations=1)
    
    # Get current alpha
    r, g, b, a = image_rgba.split()
    alpha_np = np.array(a, dtype=np.float32)
    
    # In edge zones, use more aggressive thresholding
    edge_mask = edge_zone > 0
    alpha_np[edge_mask & (alpha_np < 128)] = 0
    alpha_np[edge_mask & (alpha_np >= 128)] = 255
    
    # Smooth the result
    alpha_pil = Image.fromarray(alpha_np.astype(np.uint8))
    alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    return Image.merge('RGBA', (r, g, b, alpha_pil))

def process_multi_object_image(input_data, original_pil):
    """Process image with multiple objects - MEMORY OPTIMIZED single pass"""
    logger.info("Processing multi-object image (memory-optimized single pass)")
    
    # Single pass with general model (good for multiple objects)
    output_general = remove(
        input_data,
        session=model_loader.get_model('isnet'),  # Use lightweight isnet
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
        post_process_mask=True
    )
    
    # Open result
    img_result = Image.open(io.BytesIO(output_general)).convert('RGBA')
    
    # Clean up the mask
    r, g, b, a = img_result.split()
    alpha_np = np.array(a, dtype=np.float32)
    
    alpha_np = np.where(alpha_np < 30, 0, alpha_np)
    alpha_np = np.where(alpha_np > 225, 255, alpha_np)
    
    alpha_pil = Image.fromarray(alpha_np.astype(np.uint8))
    
    # Clean up intermediate data
    del output_general, alpha_np
    force_garbage_collection()
    
    return Image.merge('RGBA', (r, g, b, alpha_pil))

def remove_background_enhanced(original_path, job_id):
    """Enhanced background removal with better edge detection and multi-object support"""
    logger.info(f"Starting enhanced background removal for job {job_id}")
    
    with Image.open(original_path) as img:
        original_size = img.size
        original_width, original_height = original_size
        original_mode = img.mode
        
        # Store original for edge detection reference
        original_for_edges = img.copy()
        if original_for_edges.mode != 'RGB':
            original_for_edges = original_for_edges.convert('RGB')
        
        # Analyze image content to choose best approach
        original_np = np.array(original_for_edges)
        analysis = analyze_image_content(original_np)
        logger.info(f"Image analysis: human={analysis['has_human']}, product={analysis['is_product']}, logo={analysis.get('is_logo', False)}, document={analysis.get('is_document', False)}, complex={analysis['is_complex']}, uniform_bg={analysis['has_uniform_bg']}, sharp_edges={analysis['has_sharp_edges']}")
        
        # Check if this is a logo/graphic - use simplified processing
        is_logo = analysis.get('is_logo', False)
        is_document = analysis.get('is_document', False)
        
        if is_logo:
            logger.info("🎨 LOGO/GRAPHIC detected - using gentle processing to preserve all colors")
        
        if is_document:
            logger.info("📄 DOCUMENT/ID CARD detected - will preserve the entire card as foreground")
        
        # ==================== MEMORY-OPTIMIZED IMAGE SIZING ====================
        # Limit processing resolution to reduce RAM usage
        # Higher resolution = exponentially more RAM (4x resolution = 16x RAM)
        MAX_PROCESSING_DIMENSION = 1024  # Max dimension for processing (was 1500)
        MAX_UPSCALE_FACTOR = 1.5  # Max upscale factor (was 2.5)
        
        min_dimension = min(original_size)
        max_dimension = max(original_size)
        scale_factor = 1.0
        processing_size = original_size
        
        # DOWNSCALE large images to save RAM
        if max_dimension > MAX_PROCESSING_DIMENSION:
            scale_factor = MAX_PROCESSING_DIMENSION / max_dimension
            processing_size = (int(original_width * scale_factor), int(original_height * scale_factor))
            img = img.resize(processing_size, Image.Resampling.LANCZOS)
            original_for_edges = original_for_edges.resize(processing_size, Image.Resampling.LANCZOS)
            logger.info(f"Downscaled large image from {original_size} to {processing_size} (saves RAM)")
        # Only upscale very small images
        elif min_dimension < 512 and not is_logo and not is_document:
            scale_factor = min(512 / min_dimension, MAX_UPSCALE_FACTOR)
            processing_size = (int(original_width * scale_factor), int(original_height * scale_factor))
            img = img.resize(processing_size, Image.Resampling.LANCZOS)
            original_for_edges = original_for_edges.resize(processing_size, Image.Resampling.LANCZOS)
            logger.info(f"Upscaled small image from {original_size} to {processing_size}")
        
        logger.info(f"Processing at resolution: {processing_size} (memory: {get_memory_usage_mb():.0f}MB)")
        
        # Convert to bytes for rembg
        img_byte_arr = io.BytesIO()
        if img.mode == 'RGBA':
            img.save(img_byte_arr, format='PNG')
        else:
            img.convert('RGB').save(img_byte_arr, format='PNG')
        input_data = img_byte_arr.getvalue()
        
        # Choose model and parameters based on content analysis
        # BiRefNet is used as primary for ALL cases (best quality)
        # NOTE: We only store model KEYS here, actual loading happens when needed
        if is_document:
            # For documents/ID cards: Treat the ENTIRE rectangular card as foreground
            # Use edge detection to find the card boundary, not internal content
            primary_model_key = 'general'  # General model better for documents
            # Document settings - preserve the entire card
            fg_threshold = 200
            bg_threshold = 80  # Very high - only remove obvious background
            erode_size = 0     # NO erosion
            logger.info("📄 Using document-optimized settings (preserving entire card)")
        elif analysis['has_human'] and not is_document:
            # For humans: BiRefNet for best edges
            primary_model_key = 'birefnet'  # BiRefNet for best edges
            # Human-optimized settings
            fg_threshold = 250
            bg_threshold = 10
            erode_size = 10
            logger.info("🧑 Using BiRefNet model (optimized for people)")
        elif is_logo:
            # For logos/graphics: Use simple processing, NO alpha matting
            # Alpha matting can damage flat color regions
            primary_model_key = 'birefnet'
            # Logo settings - very gentle, preserve all foreground
            fg_threshold = 200  # Lower threshold = keep more
            bg_threshold = 50   # Higher threshold = be sure before removing
            erode_size = 0      # NO erosion for logos
            logger.info("🎨 Using BiRefNet with GENTLE settings (logos/graphics)")
        elif analysis['is_product'] or analysis['has_sharp_edges']:
            # For products/objects: BiRefNet for best edge detection
            primary_model_key = 'birefnet'  # Use birefnet (not massive) to save memory
            # Object-optimized settings - preserve hard edges
            fg_threshold = 240
            bg_threshold = 15
            erode_size = 5  # Smaller erode to preserve sharp edges
            logger.info("📦 Using BiRefNet (best for products/objects)")
        else:
            # General: BiRefNet
            primary_model_key = 'birefnet'
            # General settings
            fg_threshold = 245
            bg_threshold = 12
            erode_size = 8
            logger.info("🖼️ Using BiRefNet (general purpose, best quality)")
        
        # Load ONLY the primary model (lazy load - saves memory)
        primary_session = model_loader.get_model(primary_model_key)
        
        # Primary background removal
        logger.info(f"Running PRIMARY pass for job {job_id}")
        
        # For logos, disable alpha matting to preserve flat colors
        if is_logo:
            output_data = remove(
                input_data,
                session=primary_session,
                alpha_matting=False,  # NO alpha matting for logos
                post_process_mask=True
            )
        elif is_document:
            # For documents, use minimal alpha matting
            output_data = remove(
                input_data,
                session=primary_session,
                alpha_matting=False,  # NO alpha matting for documents
                post_process_mask=True
            )
        else:
            output_data = remove(
                input_data,
                session=primary_session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=fg_threshold,
                alpha_matting_background_threshold=bg_threshold,
                alpha_matting_erode_size=erode_size,
                post_process_mask=True
            )
        
        output_image = Image.open(io.BytesIO(output_data)).convert('RGBA')
        
        # For documents/ID cards, use a SIMPLE approach - just detect the outer background
        if is_document:
            logger.info("Using document-preserving approach (keeping entire card)")
            
            # For documents, the problem is AI removes parts INSIDE the card
            # Solution: Detect the BACKGROUND color from corners, only remove that
            
            h, w = original_np.shape[:2]
            corner_size = min(30, h // 10, w // 10)
            
            # Sample background color from all 4 corners
            corners = [
                original_np[:corner_size, :corner_size],           # Top-left
                original_np[:corner_size, -corner_size:],          # Top-right  
                original_np[-corner_size:, :corner_size],          # Bottom-left
                original_np[-corner_size:, -corner_size:]          # Bottom-right
            ]
            
            # Get median color from corners (background color)
            corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
            bg_color = np.median(corner_pixels, axis=0).astype(np.uint8)
            logger.info(f"Detected background color: RGB{tuple(bg_color)}")
            
            # Create mask: background is pixels similar to corner color
            # Calculate color difference from background
            color_diff = np.sqrt(np.sum((original_np.astype(np.float32) - bg_color.astype(np.float32)) ** 2, axis=2))
            
            # Threshold: pixels within 50 color units of background are BG
            # Adjust threshold based on background color variance
            bg_variance = np.std(corner_pixels)
            threshold = max(40, min(80, bg_variance * 2 + 30))
            logger.info(f"Using color threshold: {threshold:.1f}")
            
            # Background mask (1 = background, 0 = foreground)
            bg_mask = (color_diff < threshold).astype(np.uint8)
            
            # Clean up the mask - remove noise
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            
            # Open to remove small foreground noise in background
            bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel_small)
            
            # Now, only keep background that is connected to the image edges
            # This ensures we don't remove anything inside the card
            h_mask, w_mask = bg_mask.shape[:2]
            
            # Create edge-connected mask using flood fill from all edges
            edge_connected = np.zeros_like(bg_mask)
            flood_mask = np.zeros((h_mask + 2, w_mask + 2), dtype=np.uint8)
            
            # Flood fill from edges where bg_mask is 1
            temp_bg = bg_mask.copy()
            # Top edge
            for x in range(w_mask):
                if temp_bg[0, x] == 1:
                    cv2.floodFill(temp_bg, flood_mask, (x, 0), 2)
            # Bottom edge
            for x in range(w_mask):
                if temp_bg[h_mask-1, x] == 1:
                    cv2.floodFill(temp_bg, flood_mask, (x, h_mask-1), 2)
            # Left edge
            for y in range(h_mask):
                if temp_bg[y, 0] == 1:
                    cv2.floodFill(temp_bg, flood_mask, (0, y), 2)
            # Right edge
            for y in range(h_mask):
                if temp_bg[y, w_mask-1] == 1:
                    cv2.floodFill(temp_bg, flood_mask, (w_mask-1, y), 2)
            
            # Only background connected to edges should be removed
            edge_connected_bg = (temp_bg == 2).astype(np.uint8)
            
            # Foreground mask (card) = NOT edge-connected background
            card_mask = (1 - edge_connected_bg) * 255
            card_mask = card_mask.astype(np.uint8)
            
            # Smooth the edges slightly
            card_mask = cv2.GaussianBlur(card_mask, (3, 3), 0)
            card_mask = np.where(card_mask > 128, 255, 0).astype(np.uint8)
            
            # Handle scaling if we upscaled
            if processing_size != original_size:
                # Resize mask to processing size
                card_mask_pil = Image.fromarray(card_mask)
                card_mask_pil = card_mask_pil.resize(processing_size, Image.Resampling.LANCZOS)
                card_mask = np.array(card_mask_pil)
            
            # Get RGB from the output image (or original)
            r, g, b, _ = output_image.split()
            alpha_pil = Image.fromarray(card_mask)
            output_image = Image.merge('RGBA', (r, g, b, alpha_pil))
            
            # Resize back if needed
            if scale_factor != 1.0:
                output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
            
            output_byte_arr = io.BytesIO()
            output_image.save(output_byte_arr, format='PNG', optimize=True)
            result_data = output_byte_arr.getvalue()
            
            # Memory cleanup
            del output_image, card_mask, original_np, input_data
            force_garbage_collection()
            logger.info(f"Document processing complete, memory: {get_memory_usage_mb():.0f}MB")
            return result_data, 1
        
        # For logos, use simplified single-pass approach
        if is_logo:
            logger.info("Using single-pass approach for logo (preserving colors)")
            # Just use the primary result with minimal post-processing
            r, g, b, a = output_image.split()
            alpha_np = np.array(a, dtype=np.uint8)
            
            # For logos, use moderate threshold to preserve all detected content
            # But also clean up obvious noise
            alpha_np = np.where(alpha_np > 100, 255, 0).astype(np.uint8)
            
            # Fill small holes inside the logo (but don't erode edges)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            alpha_np = cv2.morphologyEx(alpha_np, cv2.MORPH_CLOSE, kernel)
            
            alpha_pil = Image.fromarray(alpha_np)
            output_image = Image.merge('RGBA', (r, g, b, alpha_pil))
            
            # Resize back if needed
            if scale_factor != 1.0:
                output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
            
            output_byte_arr = io.BytesIO()
            output_image.save(output_byte_arr, format='PNG', optimize=True)
            result_data = output_byte_arr.getvalue()
            
            # Memory cleanup
            del output_image, alpha_np, input_data
            force_garbage_collection()
            logger.info(f"Logo processing complete, memory: {get_memory_usage_mb():.0f}MB")
            return result_data, 1
        
        # ==================== MEMORY-OPTIMIZED SINGLE-PASS PROCESSING ====================
        # Use only ONE model pass to minimize RAM usage (saves ~1-2GB per image)
        # The single pass with good parameters produces excellent results
        
        logger.info("Using SINGLE-PASS processing (memory-optimized)")
        
        # Get alpha from primary result
        alpha_primary = np.array(output_image.split()[3], dtype=np.float32)
        
        # Detect multiple objects from single pass
        num_objects, labeled_array, component_sizes = detect_multiple_objects(alpha_primary.astype(np.uint8))
        logger.info(f"Detected {num_objects} significant object(s) in image")
        
        # Clean up alpha channel
        alpha_combined = alpha_primary.copy()
        
        # Clean up weak detections
        alpha_combined = np.where(alpha_combined < 30, 0, alpha_combined)
        alpha_combined = np.where(alpha_combined > 225, 255, alpha_combined)
        
        # Apply the processed alpha
        r, g, b, _ = output_image.split()
        alpha_pil = Image.fromarray(alpha_combined.astype(np.uint8))
        output_image = Image.merge('RGBA', (r, g, b, alpha_pil))
        
        # Clear intermediate data to free memory
        del alpha_primary
        del input_data
        force_garbage_collection()
        
        # Apply advanced edge refinement with product-aware settings
        is_product_image = analysis['is_product'] or analysis['has_sharp_edges']
        logger.info(f"Applying advanced edge refinement for job {job_id} (is_product={is_product_image})")
        output_image = refine_edges_advanced(output_image, edge_refinement_strength=1.0, is_product=is_product_image)
        
        # Apply GrabCut refinement for products (extra precision for hard edges)
        if is_product_image and analysis['has_uniform_bg']:
            logger.info("Applying GrabCut refinement for product edges...")
            output_image = apply_grabcut_refinement(output_image, original_for_edges)
        
        # Final cleanup pass - aggressive halo removal
        r, g, b, a = output_image.split()
        alpha_final = np.array(a, dtype=np.uint8)
        
        # Use morphological operations to clean edges - different kernels for products
        if is_product_image:
            kernel_tiny = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        else:
            kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Erode slightly to remove edge artifacts
        alpha_final = cv2.erode(alpha_final, kernel_tiny, iterations=1)
        # Dilate back to restore size
        alpha_final = cv2.dilate(alpha_final, kernel_tiny, iterations=1)
        
        # Remove small isolated regions (noise)
        alpha_final = cv2.morphologyEx(alpha_final, cv2.MORPH_OPEN, kernel_small)
        # Fill small holes
        alpha_final = cv2.morphologyEx(alpha_final, cv2.MORPH_CLOSE, kernel_small)
        
        # Apply threshold - different for products vs humans
        if is_product_image:
            # More aggressive threshold for products - cleaner edges, no semi-transparency
            alpha_final = np.where(alpha_final < 80, 0, alpha_final)
            alpha_final = np.where(alpha_final > 175, 255, alpha_final)
            # Binary threshold for remaining
            semi_transparent = (alpha_final >= 80) & (alpha_final <= 175)
            alpha_final[semi_transparent] = np.where(
                alpha_final[semi_transparent] < 128, 0, 255
            ).astype(np.uint8)
        else:
            # Standard threshold for humans - allow some edge softness
            alpha_final = np.where(alpha_final < 50, 0, alpha_final)
            alpha_final = np.where(alpha_final > 200, 255, alpha_final)
            # For remaining semi-transparent pixels (50-200), apply gradient threshold
            semi_transparent = (alpha_final >= 50) & (alpha_final <= 200)
            alpha_final[semi_transparent] = np.where(
                alpha_final[semi_transparent] < 128, 0, 255
            ).astype(np.uint8)
        
        alpha_pil = Image.fromarray(alpha_final.astype(np.uint8))
        output_image = Image.merge('RGBA', (r, g, b, alpha_pil))
        
        # Resize back to original size if we scaled
        if scale_factor != 1.0:
            output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized output back to {original_size}")
        
        # Save with high quality
        output_byte_arr = io.BytesIO()
        output_image.save(output_byte_arr, format='PNG', optimize=True)
        result_data = output_byte_arr.getvalue()
        
        # ==================== MEMORY CLEANUP ====================
        # Explicitly delete large objects and force garbage collection
        try:
            del output_image, alpha_final, alpha_pil, original_for_edges, original_np
            del r, g, b
        except:
            pass
        force_garbage_collection()
        logger.info(f"Processing complete, memory after cleanup: {get_memory_usage_mb():.0f}MB")
        
        return result_data, num_objects

def verify_background_removal(processed_image, original_image):
    """
    Cross-check and verify the quality of background removal.
    Returns quality score and issues found.
    """
    logger.info("Verifying background removal quality...")
    
    issues = []
    quality_score = 100.0
    
    # Get alpha channel
    if processed_image.mode != 'RGBA':
        return 100.0, [], processed_image
    
    r, g, b, a = processed_image.split()
    alpha_np = np.array(a, dtype=np.uint8)
    
    # Check 1: Detect remaining halos (semi-transparent pixels near edges)
    semi_transparent = (alpha_np > 30) & (alpha_np < 225)
    semi_transparent_percentage = np.sum(semi_transparent) / alpha_np.size * 100
    
    if semi_transparent_percentage > 5:
        issues.append(f"High semi-transparent pixels: {semi_transparent_percentage:.1f}%")
        quality_score -= min(semi_transparent_percentage * 2, 20)
    
    # Check 2: Detect edge roughness using contour analysis
    binary_mask = (alpha_np > 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Calculate perimeter to area ratio (higher = rougher edges)
        total_area = np.sum(binary_mask > 0)
        total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
        if total_area > 0:
            roughness = total_perimeter / np.sqrt(total_area)
            if roughness > 15:  # Threshold for rough edges
                issues.append(f"Rough edges detected (roughness: {roughness:.1f})")
                quality_score -= min((roughness - 15) * 2, 15)
    
    # Check 3: Detect isolated floating pixels (noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    noise_pixels = np.sum((binary_mask > 0) & (opened == 0))
    noise_percentage = noise_pixels / alpha_np.size * 100
    
    if noise_percentage > 0.5:
        issues.append(f"Floating pixels detected: {noise_percentage:.2f}%")
        quality_score -= min(noise_percentage * 5, 10)
    
    # Check 4: Verify foreground isn't too small (possible over-removal)
    foreground_percentage = np.sum(alpha_np > 128) / alpha_np.size * 100
    if foreground_percentage < 5:
        issues.append(f"Foreground very small: {foreground_percentage:.1f}%")
        quality_score -= 20
    elif foreground_percentage < 15:
        issues.append(f"Foreground small: {foreground_percentage:.1f}%")
        quality_score -= 10
    
    # Check 5: Compare with original to detect lost details
    original_gray = np.array(original_image.convert('L'))
    if original_gray.shape == alpha_np.shape:
        # Detect high-contrast areas in original that might be lost
        edges_original = cv2.Canny(original_gray, 50, 150)
        edges_masked = edges_original.copy()
        edges_masked[alpha_np < 128] = 0
        edge_preservation = np.sum(edges_masked > 0) / max(np.sum(edges_original > 0), 1) * 100
        
        if edge_preservation < 60:
            issues.append(f"Possible detail loss: {100-edge_preservation:.1f}%")
            quality_score -= min((60 - edge_preservation) * 0.5, 15)
    
    quality_score = max(0, min(100, quality_score))
    logger.info(f"Quality score: {quality_score:.1f}%, Issues: {issues}")
    
    return quality_score, issues, processed_image

def fix_quality_issues(processed_image, original_image, issues):
    """
    Attempt to fix detected quality issues.
    """
    logger.info(f"Attempting to fix quality issues: {issues}")
    
    if processed_image.mode != 'RGBA':
        return processed_image
    
    r, g, b, a = processed_image.split()
    alpha_np = np.array(a, dtype=np.uint8)
    
    # Fix semi-transparent halos
    if any('semi-transparent' in issue.lower() for issue in issues):
        logger.info("Fixing semi-transparent pixels...")
        # Aggressive threshold
        alpha_np = np.where(alpha_np < 80, 0, alpha_np)
        alpha_np = np.where(alpha_np > 175, 255, alpha_np)
        # Hard threshold remaining
        alpha_np = np.where((alpha_np >= 80) & (alpha_np <= 175), 
                           np.where(alpha_np < 128, 0, 255), alpha_np).astype(np.uint8)
    
    # Fix floating pixels / noise
    if any('floating' in issue.lower() or 'noise' in issue.lower() for issue in issues):
        logger.info("Removing floating pixels...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_np = cv2.morphologyEx(alpha_np, cv2.MORPH_OPEN, kernel)
    
    # Fix rough edges
    if any('rough' in issue.lower() for issue in issues):
        logger.info("Smoothing rough edges...")
        # Apply morphological closing then opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_np = cv2.morphologyEx(alpha_np, cv2.MORPH_CLOSE, kernel)
        alpha_np = cv2.morphologyEx(alpha_np, cv2.MORPH_OPEN, kernel)
        # Slight blur for smoothness
        alpha_np = cv2.GaussianBlur(alpha_np, (3, 3), 0.5)
        alpha_np = np.where(alpha_np < 128, 0, 255).astype(np.uint8)
    
    alpha_pil = Image.fromarray(alpha_np)
    return Image.merge('RGBA', (r, g, b, alpha_pil))

def enhance_final_image(processed_image, enhancement_level='medium'):
    """
    Enhance the final processed image for better visual quality.
    
    Args:
        processed_image: PIL Image with transparent background
        enhancement_level: 'light', 'medium', or 'strong'
    
    Returns:
        Enhanced PIL Image
    """
    logger.info(f"Enhancing final image with {enhancement_level} level...")
    
    if processed_image.mode != 'RGBA':
        processed_image = processed_image.convert('RGBA')
    
    # Split into RGB and Alpha
    r, g, b, a = processed_image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    # Enhancement parameters based on level
    params = {
        'light': {'sharpness': 1.1, 'contrast': 1.02, 'color': 1.02, 'brightness': 1.0},
        'medium': {'sharpness': 1.2, 'contrast': 1.05, 'color': 1.05, 'brightness': 1.02},
        'strong': {'sharpness': 1.4, 'contrast': 1.1, 'color': 1.1, 'brightness': 1.05}
    }
    
    p = params.get(enhancement_level, params['medium'])
    
    # Step 1: Sharpen the image
    enhancer = ImageEnhance.Sharpness(rgb_image)
    rgb_image = enhancer.enhance(p['sharpness'])
    
    # Step 2: Slight contrast boost
    enhancer = ImageEnhance.Contrast(rgb_image)
    rgb_image = enhancer.enhance(p['contrast'])
    
    # Step 3: Subtle color enhancement
    enhancer = ImageEnhance.Color(rgb_image)
    rgb_image = enhancer.enhance(p['color'])
    
    # Step 4: Brightness adjustment (very subtle)
    enhancer = ImageEnhance.Brightness(rgb_image)
    rgb_image = enhancer.enhance(p['brightness'])
    
    # Step 5: Apply unsharp mask for crisp edges (using numpy/opencv)
    rgb_np = np.array(rgb_image)
    
    # Unsharp mask: original + (original - blurred) * amount
    blurred = cv2.GaussianBlur(rgb_np, (0, 0), 2.0)
    sharpened = cv2.addWeighted(rgb_np, 1.3, blurred, -0.3, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    rgb_image = Image.fromarray(sharpened)
    
    # Step 6: Clean up alpha channel edges
    alpha_np = np.array(a, dtype=np.uint8)
    
    # Detect edges of the mask
    edges = cv2.Canny(alpha_np, 100, 200)
    edge_zone = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    # Apply slight feathering only at edges for natural look
    alpha_float = alpha_np.astype(np.float32)
    blurred_alpha = cv2.GaussianBlur(alpha_float, (3, 3), 0.5)
    
    # Blend at edges only
    edge_mask = (edge_zone > 0).astype(np.float32) * 0.3
    alpha_result = alpha_float * (1 - edge_mask) + blurred_alpha * edge_mask
    alpha_result = np.clip(alpha_result, 0, 255).astype(np.uint8)
    
    # Re-threshold to keep crisp edges
    alpha_result = np.where(alpha_result < 30, 0, alpha_result)
    alpha_result = np.where(alpha_result > 225, 255, alpha_result)
    
    alpha_pil = Image.fromarray(alpha_result)
    
    # Merge back with enhanced RGB
    r, g, b = rgb_image.split()
    enhanced_image = Image.merge('RGBA', (r, g, b, alpha_pil))
    
    logger.info("Image enhancement complete")
    return enhanced_image

def process_with_verification_and_enhancement(original_path, job_id):
    """
    Complete pipeline: Remove background -> Verify -> Fix issues -> Enhance
    """
    logger.info(f"Starting complete processing pipeline for job {job_id}")
    
    # Step 1: Remove background
    processed_data, num_objects = remove_background_enhanced(original_path, job_id)
    processed_image = Image.open(io.BytesIO(processed_data)).convert('RGBA')
    
    # Step 2: Load original for verification
    original_image = Image.open(original_path)
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    
    # Resize original to match processed if needed
    if original_image.size != processed_image.size:
        original_image = original_image.resize(processed_image.size, Image.Resampling.LANCZOS)
    
    # Step 3: Verify quality
    quality_score, issues, processed_image = verify_background_removal(processed_image, original_image)
    logger.info(f"Initial quality score: {quality_score:.1f}%")
    
    # Step 4: Fix issues if quality is below threshold
    if quality_score < 85 and issues:
        logger.info("Quality below threshold, attempting fixes...")
        processed_image = fix_quality_issues(processed_image, original_image, issues)
        
        # Re-verify after fixes
        quality_score, issues, processed_image = verify_background_removal(processed_image, original_image)
        logger.info(f"Quality after fixes: {quality_score:.1f}%")
    
    # Step 5: Enhance the final image
    # Choose enhancement level based on quality
    if quality_score >= 90:
        enhancement_level = 'light'
    elif quality_score >= 75:
        enhancement_level = 'medium'
    else:
        enhancement_level = 'strong'
    
    enhanced_image = enhance_final_image(processed_image, enhancement_level)
    
    # Step 6: Save the final result
    output_byte_arr = io.BytesIO()
    enhanced_image.save(output_byte_arr, format='PNG', optimize=True)
    
    logger.info(f"Processing complete for job {job_id}. Quality: {quality_score:.1f}%, Objects: {num_objects}")
    
    return output_byte_arr.getvalue(), num_objects, quality_score, issues

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
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
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

def secure_filename(filename):
    """Custom secure filename function to handle special characters"""
    # Get file extension
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        name = filename.rsplit('.', 1)[0]
        # Replace special characters
        name = ''.join(c for c in name if c.isalnum() or c in ['-', '_'])
        if not name:
            name = 'unnamed'
        return f"{name}.{ext}"
    return 'unnamed.png'

@app.route('/')
def index():
    """Render the main landing page"""
    return render_template('index.html')

# ==================== ROUTE 1: UPLOAD ====================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('bgremove.html')
    
    elif request.method == 'POST':
        try:
            # Check if file exists in request
            logger.info(f"POST request received to /upload with files: {list(request.files.keys())}")
            
            if 'image' not in request.files:
                logger.error("No file part in request")
                return jsonify({
                    'success': False,
                    'error': 'No file part',
                    'step': 'upload'
                }), 400
            
            file = request.files['image']
            
            # Check if filename is empty
            if file.filename == '':
                logger.error("No file selected")
                return jsonify({
                    'success': False,
                    'error': 'No file selected',
                    'step': 'upload'
                }), 400
            
            # Generate a unique job ID
            job_id = str(uuid.uuid4())
            logger.info(f"Generated new job ID: {job_id}")
            
            # Create job directories
            job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
            originals_dir = os.path.join(job_dir, 'originals')
            processed_dir = os.path.join(job_dir, 'processed')
            
            os.makedirs(originals_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            
            # Secure the filename
            original_filename = secure_filename(file.filename)
            
            # Save the original file
            original_path = os.path.join(originals_dir, original_filename)
            file.save(original_path)
            logger.info(f"Saved original file to: {original_path}")
            
            # Save job info
            job_info = {
                'job_id': job_id,
                'status': 'uploaded',
                'user_filename': file.filename,
                'original_filename': original_filename,
                'original_path': original_path,
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'original_url': f"/file/originals/{job_id}/{original_filename}",
                    'user_filename': file.filename
                }
            }
            
            if not save_job_info(job_id, job_info):
                return jsonify({
                    'success': False,
                    'error': 'Failed to save job information',
                    'step': 'upload'
                }), 500
            
            # Create URLs for preview
            original_url = f"/file/originals/{job_id}/{original_filename}"
            
            logger.info(f"Upload completed for job {job_id}")
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'original_url': original_url,
                'filename': file.filename,
                'step': 'upload_complete',
                'message': 'Image uploaded successfully',
                'next_step': '/initiate-process',
                'redirect_url': f'/initiate-process?job_id={job_id}',
                'auto_refresh': True
            })
            
        except Exception as e:
            logger.error(f"Error in upload: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e),
                'step': 'upload'
            }), 500

# ==================== ROUTE 2: INITIATE PROCESS ====================
@app.route('/initiate-process', methods=['GET', 'POST'])
def initiate_process():
    """STEP 2: Start the background removal processing"""
    if request.method == 'GET':
        # Handle GET request with refresh
        job_id = request.args.get('job_id')
        if not job_id:
            return redirect(url_for('upload'))
        
        # Load job info to verify it exists
        job_info = load_job_info(job_id)
        if not job_info:
            return redirect(url_for('upload'))
        
        # Return the processing page with refresh headers
        response = make_response(render_template('bgremove.html', job_id=job_id, step='initiate_process'))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
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
            job_info['process_start_time'] = datetime.now().isoformat()
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
                'redirect_url': f'/background-removed?job_id={job_id}',
                'auto_refresh': True  # Add this flag
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
        # Handle GET request with refresh
        job_id = request.args.get('job_id')
        if not job_id:
            return redirect(url_for('upload'))
        
        # Load job info to verify it exists
        job_info = load_job_info(job_id)
        if not job_info:
            return redirect(url_for('upload'))
        
        # Return the results page with refresh headers
        response = make_response(render_template('bgremove.html', job_id=job_id, step='background_removed'))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
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
                        'redirect_url': f'/background-removed?job_id={job_id}&completed=true',
                        'auto_refresh': True  # Add this flag
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
            
            # NOW PERFORM THE ACTUAL BACKGROUND REMOVAL USING ENHANCED FUNCTION
            logger.info(f"Starting enhanced background removal with verification for job {job_id}")
            
            original_path = job_info['original_path']
            processed_path = job_info['processed_path']
            
            try:
                # Use the complete pipeline with verification and enhancement
                output_data, num_objects, quality_score, issues = process_with_verification_and_enhancement(original_path, job_id)
                
                logger.info(f"Complete processing done for job {job_id} - {num_objects} object(s), quality: {quality_score:.1f}%")
                
                # Save the result
                with open(processed_path, 'wb') as output_file:
                    output_file.write(output_data)
                
                # Update status to completed
                job_info['status'] = 'completed'
                job_info['process_end_time'] = datetime.now().isoformat()
                job_info['step'] = 'background_removed'
                job_info['objects_detected'] = num_objects
                job_info['quality_score'] = quality_score
                job_info['quality_issues'] = issues
                
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
                        'message': f'Background removal completed successfully - {num_objects} object(s) detected',
                        'data': {
                            'filename': job_info['processed_filename'],
                            'original_url': original_url,
                            'processed_url': processed_url,
                            'original_data': f"data:{original_mime};base64,{original_base64}",
                            'processed_data': f"data:{processed_mime};base64,{processed_base64}",
                            'user_filename': job_info['user_filename'],
                            'objects_detected': num_objects
                        },
                        'redirect_url': f'/background-removed?job_id={job_id}&completed=true',
                        'auto_refresh': True
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
                import traceback
                logger.error(traceback.format_exc())
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
@app.route('/file/<path:filename>')
def serve_file(filename):
    """Serve files from upload folder"""
    # For security, validate that the filename is within allowed paths
    if ".." in filename or filename.startswith("/"):
        abort(404)
    
    # For originals files: /file/originals/job_id/filename
    if filename.startswith('originals/'):
        parts = filename.split('/')
        if len(parts) >= 3:  # originals/job_id/filename
            job_id = parts[1]
            file_name = parts[2]
            return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], job_id, 'originals'), file_name)
    
    # For processed files: /file/processed/filename (in shared processed folder)
    if filename.startswith('processed/'):
        parts = filename.split('/')
        if len(parts) >= 2:  # processed/filename
            file_name = parts[1]
            return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'processed'), file_name)
    
    abort(404)

@app.route('/download/<filename>')
def download_file(filename):
    """Handle file downloads with proper headers"""
    directory = os.path.join(app.config['UPLOAD_FOLDER'], 'processed')
    response = send_from_directory(directory, filename, as_attachment=True)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ==================== TASK PAGE ROUTES ====================
@app.route('/taskpage1.html')
def taskpage1():
    """Route to task page 1 - Preview Quality Download"""
    return render_template('taskpage1.html')

@app.route('/taskpage2.html') 
def taskpage2():
    """Route to task page 2 - Original Quality Download"""
    return render_template('taskpage2.html')

@app.route('/bgremove')
def bgremove():
    """Alternative route to main background remover"""
    return redirect(url_for('upload'))

# ==================== ROUTE: RESULT PAGE ====================
@app.route('/result')
def result_page():
    """Display the result page with processed images"""
    job_id = request.args.get('job_id')
    
    if not job_id:
        return redirect(url_for('upload'))
    
    # Load job info
    job_info = load_job_info(job_id)
    
    if not job_info:
        return redirect(url_for('upload'))
    
    # Check if job is completed
    if job_info.get('status') != 'completed':
        # If not completed, redirect to process
        return redirect(url_for('initiate_process', job_id=job_id))
    
    # Prepare data for template
    original_url = f"/file/originals/{job_id}/{job_info['original_filename']}"
    processed_url = f"/file/processed/{job_info['processed_filename']}"
    download_url = f"/download/{job_info['processed_filename']}"
    
    return render_template('result.html', 
                         job_id=job_id,
                         original_url=original_url,
                         processed_url=processed_url,
                         download_url=download_url,
                         filename=job_info.get('user_filename', 'image.png'))

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
# Auto-cleanup settings: Delete files older than 10 minutes
CLEANUP_INTERVAL_SECONDS = 60  # Check every minute
FILE_RETENTION_MINUTES = 10   # Delete files after 10 minutes

cleanup_thread = None
cleanup_running = False

def cleanup_old_files_background():
    """Background thread to clean up old files automatically"""
    global cleanup_running
    cleanup_running = True
    
    while cleanup_running:
        try:
            now = datetime.now()
            retention_period = timedelta(minutes=FILE_RETENTION_MINUTES)
            upload_folder = app.config['UPLOAD_FOLDER']
            
            # Clean up job directories (each upload creates a job folder)
            if os.path.exists(upload_folder):
                for item in os.listdir(upload_folder):
                    item_path = os.path.join(upload_folder, item)
                    
                    # Skip non-directories and special folders
                    if not os.path.isdir(item_path):
                        continue
                    if item in ['originals', 'processed', 'jobs']:
                        # Clean up legacy folders
                        for filename in os.listdir(item_path):
                            filepath = os.path.join(item_path, filename)
                            try:
                                file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                                if (now - file_modified) > retention_period:
                                    if os.path.isdir(filepath):
                                        shutil.rmtree(filepath)
                                    else:
                                        os.remove(filepath)
                                    logger.info(f"Cleaned up old file: {filepath}")
                            except Exception as e:
                                logger.error(f"Error cleaning file {filepath}: {str(e)}")
                        continue
                    
                    # Check if job directory is old enough to delete
                    try:
                        # Check the modification time of the directory
                        dir_modified = datetime.fromtimestamp(os.path.getmtime(item_path))
                        if (now - dir_modified) > retention_period:
                            # Delete the entire job directory
                            shutil.rmtree(item_path)
                            logger.info(f"Cleaned up old job directory: {item_path}")
                            
                            # Also remove from in-memory status
                            if item in processing_status:
                                del processing_status[item]
                    except Exception as e:
                        logger.error(f"Error cleaning directory {item_path}: {str(e)}")
            
            # Clean up expired jobs from in-memory storage
            expired_jobs = []
            for job_id, job_info in list(processing_status.items()):
                try:
                    if 'upload_time' in job_info:
                        job_time = datetime.fromisoformat(job_info['upload_time'])
                        if (now - job_time) > retention_period:
                            expired_jobs.append(job_id)
                except Exception as e:
                    logger.error(f"Error checking job {job_id}: {str(e)}")
            
            for job_id in expired_jobs:
                del processing_status[job_id]
                logger.info(f"Removed expired job from memory: {job_id}")
                
        except Exception as e:
            logger.error(f"Error in cleanup thread: {str(e)}")
        
        # Sleep for the cleanup interval
        time.sleep(CLEANUP_INTERVAL_SECONDS)

def start_cleanup_thread():
    """Start the background cleanup thread"""
    global cleanup_thread
    if cleanup_thread is None or not cleanup_thread.is_alive():
        cleanup_thread = threading.Thread(target=cleanup_old_files_background, daemon=True)
        cleanup_thread.start()
        logger.info(f"Started automatic cleanup thread (deleting files older than {FILE_RETENTION_MINUTES} minutes)")

def stop_cleanup_thread():
    """Stop the background cleanup thread"""
    global cleanup_running
    cleanup_running = False
    logger.info("Stopped cleanup thread")

# Register cleanup on exit
atexit.register(stop_cleanup_thread)

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/memory-status')
def memory_status():
    """Endpoint to check memory usage and loaded models"""
    try:
        memory_mb = get_memory_usage_mb()
        loaded_models = model_loader.get_loaded_models()
        return jsonify({
            'status': 'ok',
            'memory_usage_mb': round(memory_mb, 1),
            'loaded_models': loaded_models,
            'model_count': len(loaded_models)
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/unload-models', methods=['POST'])
def unload_models():
    """Endpoint to manually unload all models and free memory"""
    try:
        mem_before = get_memory_usage_mb()
        model_loader.unload_all()
        force_garbage_collection()
        mem_after = get_memory_usage_mb()
        return jsonify({
            'status': 'ok',
            'memory_before_mb': round(mem_before, 1),
            'memory_after_mb': round(mem_after, 1),
            'memory_freed_mb': round(mem_before - mem_after, 1)
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# Start cleanup thread when module is imported (for production servers like gunicorn)
start_cleanup_thread()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() in ['true', '1', 't']
    
    logger.info(f"Starting application on port {port} with debug={debug}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Auto-cleanup enabled: files older than {FILE_RETENTION_MINUTES} minutes will be deleted")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
