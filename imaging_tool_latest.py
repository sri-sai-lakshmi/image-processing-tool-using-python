from webbrowser import Opera
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from PIL import Image
import io
import base64
import os
import traceback
import cv2
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
from werkzeug.utils import secure_filename
#from volume_vedo import fuse_and_display_volume

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

original_image = None
current_image = None
image_format = None
ct_image = None
mri_image = None
original_ct_image = None
original_mri_image = None
original_blend_method = None
original_blend_alpha = 0.5


extensions = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def load_dicom(filepath):
    ds=pydicom.dcmread(filepath)
    img_array=ds.pixel_array.astype(np.float32)
    img_array -= np.min(img_array)
    img_array /= np.max(img_array)
    img_array *= 255
    return img_array.astype(np.uint8)

def register_mri_to_ct(ct_array, mri_array):
    # Convert NumPy arrays to SimpleITK images
    ct = sitk.GetImageFromArray(ct_array.astype(np.float32))
    mri = sitk.GetImageFromArray(mri_array.astype(np.float32))

    # Only allow translation — no rotation
    transform = sitk.TranslationTransform(2)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetInitialTransform(transform)
    registration_method.SetInterpolator(sitk.sitkLinear)

    final_transform = registration_method.Execute(ct, mri)

    aligned_mri = sitk.Resample(mri, ct, final_transform, sitk.sitkLinear, 0.0, mri.GetPixelID())

    return sitk.GetArrayFromImage(aligned_mri).astype(np.uint8)

def rgb_to_grayscale(image_array):
    #check if the image is rgb , if yes then extract the rgb channel and multiple and
    # clip it to int values, if alredy grayscale then just return
    if len(image_array.shape) == 3:
        #luminance weights: 0.299*R + 0.587*G + 0.114*B
        grayscale = np.dot(image_array[...,:3], [0.299, 0.587, 0.114])
        return grayscale.astype(np.uint8)
    return image_array

def resize_image(image_array, new_width, new_height):
   
    old_height, old_width = image_array.shape[:2]
    
    # Calculate scaling factor i.e upscaling or downscaling
    x_scale = old_width / new_width
    y_scale = old_height / new_height
    
    # here we make all the coordinates of the new size of the image
    new_x = np.arange(new_width)
    new_y = np.arange(new_height)
    
    # Map new coordinates to old coordinates using nearset neighbor interpolation
    old_x = (new_x * x_scale).astype(int)
    old_y = (new_y * y_scale).astype(int)
    
    # Clip to ensure indices are in bound
    old_x = np.clip(old_x, 0, old_width - 1)
    old_y = np.clip(old_y, 0, old_height - 1)
    
    # Create meshgrid for indexing
    old_y_mesh, old_x_mesh = np.meshgrid(old_y, old_x, indexing='ij')
    
    # Resize the image
    if len(image_array.shape) == 3:
        resized = image_array[old_y_mesh, old_x_mesh, :]
    else:
        resized = image_array[old_y_mesh, old_x_mesh]
    
    return resized

def crop_image(image_array, x, y, width, height):
    img_height, img_width = image_array.shape[:2]
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    width = min(width, img_width - x)
    height = min(height, img_height - y)
    return image_array[y:y+height, x:x+width]

#def adjust_brightness(image_array, factor):
    
# Convert to float to prevent overflow
    #adjusted = image_array.astype(np.float32) * factor
    # Clip values to valid range and convert back to uint8
    #adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    #return adjusted

def adjust_brightness(image_array, factor):
    image_array = image_array.astype(np.float32)

    if factor == 0:
        return np.zeros_like(image_array, dtype=np.uint8)
    elif factor == 1:
        return image_array.astype(np.uint8)
    elif factor > 1:
        # Blend towards white
        adjusted = image_array + (255 - image_array) * (factor - 1)
    else:
        # Blend toward black
        adjusted = image_array * factor

    return np.clip(adjusted, 0, 255).astype(np.uint8)


def invert_colors(image_array):
    return 255 - image_array

def rotate_image(image_array, angle):
    pil_image = Image.fromarray(image_array)
    rotated = pil_image.rotate(angle, expand=True)  # expand=True avoids cropping
    return np.array(rotated)

def mirror_image(image_array):
    height,width=image_array.shape[:2]
    image_flip=np.zeros_like(image_array)
    if len(image_array.shape) == 2: #grayscale
        for y in range(height):
            for x in range(width):
                image_flip[y,x]=image_array[y,width-x-1]
    else:
        for y in range(height):
            for x in range(width):
                image_flip[y,x]=image_array[y,width-x-1]
    return image_flip           
   
def apply_rgba_alpha(image_array, alpha):
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    if len(image_array.shape) == 3:
        alpha_channel = np.full((image_array.shape[0], image_array.shape[1], 1), int(alpha * 255), dtype=np.uint8)
        rgba_image = np.concatenate((image_array, alpha_channel), axis=-1)
        return rgba_image
    else:
        rgb = np.stack([image_array]*3, axis=-1)
        alpha_channel = np.full((image_array.shape[0], image_array.shape[1], 1), int(alpha * 255), dtype=np.uint8)
        return np.concatenate((rgb, alpha_channel), axis=-1)
    
def numpy_to_base64(image_array, format_type):    
    if len(image_array.shape) == 2:       
        image_array = np.stack([image_array] * 3, axis=-1)    
    pil_image = Image.fromarray(image_array)
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format_type.upper())
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format_type.lower()};base64,{img_base64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display_volume', methods=['POST'])
def display_volume():
    ct_folder = r"C:\Users\z005563b\Downloads\CT_folder"
    mri_folder = r"C:\Users\z005563b\Downloads\AID_3003000 - Copy\AID_3003000 - Copy\MR.001"

    fuse_and_display_volume(ct_folder, mri_folder)

    return "3D Volume viewer opened with fused data."

@app.route('/upload', methods=['POST'])
def upload_file():
    global original_image, current_image, image_format

    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_format = filename.rsplit('.', 1)[1].lower()
        if image_format == 'jpg':
            image_format = 'jpeg'
        try:
            pil_image = Image.open(file.stream)
            pil_image = pil_image.convert('RGB')
            original_image = np.array(pil_image)
            print(f"Uploaded image shape: {original_image.shape}")

            current_image = original_image.copy()

            image_b64 = numpy_to_base64(current_image, image_format)
            print(f"Returning image of format: {image_format}")

            return jsonify({
                'success': True,
                'image': image_b64,
                'width': current_image.shape[1],
                'height': current_image.shape[0]
            })
        except Exception as e:
            return jsonify({'error': f'Failed to load image: {str(e)}'}), 400

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/upload_dicom', methods=['POST'])
def upload_dicom():
    global ct_image, mri_image, current_image, image_format

    if 'ct' not in request.files or 'mri' not in request.files:
        return jsonify({'error': 'Both CT and MRI files must be provided'}), 400

    ct_file = request.files['ct']
    mri_file = request.files['mri']

    ct_image = load_dicom(ct_file)
    mri_image = load_dicom(mri_file)

    # Resize both to smallest common size
    h = min(ct_image.shape[0], mri_image.shape[0])
    w = min(ct_image.shape[1], mri_image.shape[1])
    ct_image = resize_image(ct_image, w, h)
    mri_image = resize_image(mri_image, w, h)
    mri_image = register_mri_to_ct(ct_image, mri_image)
    
    original_ct_image = ct_image.copy()
    original_mri_image = mri_image.copy()

    # Preview using alpha blend by default
        # Generate previews separately
    ct_preview = numpy_to_base64(ct_image, 'png')
    mri_preview = numpy_to_base64(mri_image, 'png')
    current_image=ct_image.copy()
    original_image = current_image.copy()  
    image_format='png'
    
    return jsonify({
        'success': True,
        'ct_image': ct_preview,
        'mri_image': mri_preview,
        'width': ct_image.shape[1],
        'height': ct_image.shape[0]
    })
    

@app.route('/process', methods=['POST'])
def process_image():
    global current_image, image_format, ct_image, mri_image
    global original_image, original_blend_method, original_blend_alpha
    
    data = request.json
    operation = data.get('operation')
    params = data.get('params', {})
    
    if current_image is None and operation != 'blend':
        return jsonify({'error': 'No image uploaded'}), 400
    
    try:
        if operation == 'grayscale':
            current_image = rgb_to_grayscale(current_image)
        
        elif operation == 'resize':
            width = int(params.get('width', current_image.shape[1]))
            height = int(params.get('height', current_image.shape[0]))
            current_image = resize_image(current_image, width, height)
        
        elif operation == 'crop':
            x = int(params.get('x', 0))
            y = int(params.get('y', 0))
            width = int(params.get('width', current_image.shape[1]))
            height = int(params.get('height', current_image.shape[0]))
            current_image = crop_image(current_image, x, y, width, height)
        
        elif operation == 'brighten':
            factor = float(params.get('factor', 1.2))
            current_image = adjust_brightness(current_image, factor)
        
        elif operation == 'invert':
            current_image = invert_colors(current_image)

        elif operation == 'rotate':
            angle = int(params.get('angle', 0))  
            current_image = rotate_image(current_image, angle)
        elif operation == 'mirror':
            current_image=mirror_image(current_image)
        elif operation == 'rgba_alpha':
            alpha_val=float(params.get('alpha',1.0))
            current_image=apply_rgba_alpha(current_image,alpha_val)
        elif operation == 'blend':
            method = params.get('method', 'alpha')
            alpha = float(params.get('alpha', 0.5))

            if ct_image is None or mri_image is None:
                return jsonify({'error': 'CT or MRI image missing'}), 400

            if ct_image.shape != mri_image.shape:
                return jsonify({'error': 'Image dimensions do not match'}), 400

            if method == 'alpha':
                current_image = (ct_image * alpha + mri_image * (1 - alpha)).astype(np.uint8)
            elif method == 'multiply':
                current_image = ((ct_image.astype(np.float32) / 255.0) *
                                 (mri_image.astype(np.float32) / 255.0) * 255.0)
                current_image = np.clip(current_image, 0, 255).astype(np.uint8)
            elif method == 'false_color':
                ct = ct_image.astype(np.float32) / 255.0
                mri = mri_image.astype(np.float32) / 255.0
                rgb_image = np.zeros((ct.shape[0], ct.shape[1], 3), dtype=np.float32)
                rgb_image[:, :, 0] = ct
                rgb_image[:, :, 1] = mri
                rgb_image[:, :, 2] = 0
                current_image = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
            

            else:
                return jsonify({'error': 'Invalid blending method'}), 400
            original_image = current_image.copy()
            original_blend_method = method
            original_blend_alpha = alpha

            #print("Current image shape after blend:", current_image.shape)
            #return jsonify({
            #    'success': True,
            #    'blend_processed': True,
            #    'method': method,
            #    'alpha': alpha if method == 'alpha' else None
            #})
        
            if len(current_image.shape) == 3 and current_image.shape[-1] == 4:
                image_format = 'png'

            image_b64 = numpy_to_base64(current_image, image_format)
            ct_b64 = numpy_to_base64(ct_image, 'png')
            mri_b64 = numpy_to_base64(mri_image, 'png')
        
            return jsonify({
                'success': True,
                'image': image_b64,
                'ct_image': ct_b64,
                'mri_image': mri_b64,
                'width': current_image.shape[1],
                'height': current_image.shape[0]
            })
        image_b64 = numpy_to_base64(current_image, image_format)

        return jsonify({
            'success': True,
            'image': image_b64,
            'width': current_image.shape[1],
            'height': current_image.shape[0]
        })
    except Exception as e:
        print("Exception in /process:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/reset', methods=['POST'])
def reset_image():
    global current_image, original_image, image_format
    global ct_image, mri_image, original_blend_method, original_blend_alpha

    print(" /reset endpoint hit")
    print(f" ct_image: {ct_image is not None}, mri_image: {mri_image is not None}")
    print(f" original_blend_method: {original_blend_method}, alpha: {original_blend_alpha}")
    print(f" original_image is None? {original_image is None}")

    # Case 1: DICOM blend exists
    if ct_image is not None and mri_image is not None and original_blend_method:
        print(" Reconstructing DICOM blend...")
        method = original_blend_method
        alpha = original_blend_alpha

        try:
            if method == 'alpha':
                current_image = (ct_image * alpha + mri_image * (1 - alpha)).astype(np.uint8)
            elif method == 'multiply':
                current_image = ((ct_image.astype(np.float32) / 255.0) *
                                 (mri_image.astype(np.float32) / 255.0) * 255.0)
                current_image = np.clip(current_image, 0, 255).astype(np.uint8)
            elif method == 'false_color':
                ct = ct_image.astype(np.float32) / 255.0
                mri = mri_image.astype(np.float32) / 255.0
                rgb_image = np.zeros((ct.shape[0], ct.shape[1], 3), dtype=np.float32)
                rgb_image[:, :, 0] = ct
                rgb_image[:, :, 1] = mri
                rgb_image[:, :, 2] = 0
                current_image = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
            else:
                return jsonify({'error': 'Unknown blend method'}), 400

            original_image = current_image.copy()  # sync for further reset
            print("✅ Blend reconstructed and state updated.")

        except Exception as e:
            print("❌ Exception during blend reconstruction:", e)
            traceback.print_exc()
            return jsonify({'error': f"Blend reconstruction failed: {str(e)}"}), 500

    # Case 2: Regular image reset
    elif original_image is not None:
        print("🌀 Resetting regular image (non-DICOM)")
        current_image = original_image.copy()

    # Case 3: Nothing to reset
    else:
        print("❌ No image uploaded — cannot reset.")
        return jsonify({'error': 'No image uploaded'}), 400

    # Return reset image
    image_b64 = numpy_to_base64(current_image, image_format)

    return jsonify({
        'success': True,
        'image': image_b64,
        'width': current_image.shape[1],
        'height': current_image.shape[0]
    })




@app.route('/download')
def download_image():
    global current_image, image_format
    
    if current_image is None:
        return jsonify({'error': 'No image to download'}), 400
    
    # Handle grayscale images for PIL
    if len(current_image.shape) == 2:
        # Convert grayscale to RGB for PIL
        image_for_pil = np.stack([current_image] * 3, axis=-1)
    if current_image.shape[-1] == 4:
        image_format = 'png'
        image_for_pil = current_image

    else:
        image_for_pil = current_image
    
    # Create PIL image
    pil_image = Image.fromarray(image_for_pil)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format=image_format.upper())
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'processed_image.{image_format}',
        mimetype=f'image/{image_format}'
    )



if __name__ == '__main__':
    app.run(debug=True)

