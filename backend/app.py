from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

# Folder for uploading images and saving heatmap images
UPLOAD_FOLDER = 'static/images'
HEATMAP_FOLDER = 'static/heatmaps'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER

# Create the directories if they do not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

def calculate_vegetation_index(image):
    # Convert the image to RGB (if it's not already)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract Red, Green, and Blue channels
    red = image_rgb[:, :, 0].astype(float)
    green = image_rgb[:, :, 1].astype(float)
    blue = image_rgb[:, :, 2].astype(float)
    
    # Calculate VARI (Vegetation Index)
    vari = (green - red) / (green + red - blue + 1e-6)  # Adding epsilon to avoid division by zero

    # Normalize the VARI values for better visualization (scale to [0, 255])
    vari_normalized = cv2.normalize(vari, None, 0, 255, cv2.NORM_MINMAX)

    return vari_normalized

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Save the uploaded file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Read the image using OpenCV
        image = cv2.imread(filepath)
        
        # Process the image and get the VARI heatmap
        vari = calculate_vegetation_index(image)
        
        # Apply color map to VARI values to create a heatmap
        vari_colormap = cv2.applyColorMap(vari.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Save the heatmap as an image file
        vari_heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], f'{file.filename}_vari_heatmap.png')
        cv2.imwrite(vari_heatmap_path, vari_colormap)
        
        # Send the heatmap and original image to the front-end
        return render_template('index.html', 
                               original_image=file.filename, 
                               vari_image=f'{file.filename}_vari_heatmap.png')

if __name__ == '__main__':
    app.run(debug=True)
