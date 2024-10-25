# app.py

import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from rembg import remove
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the original uploaded file
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)

        # Perform background removal
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'bg_removed_' + file.filename)
        with open(input_path, 'rb') as input_file:
            input_data = input_file.read()
        
        output_data = remove(input_data)
        
        # Save the result
        with open(output_path, 'wb') as output_file:
            output_file.write(output_data)

        return redirect(url_for('result', filename='bg_removed_' + file.filename))

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
