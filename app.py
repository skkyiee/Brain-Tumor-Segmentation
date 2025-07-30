import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import cv2

app = Flask(__name__)

# Make upload path absolute
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = os.path.join(BASE_DIR, 'final_optimized_brain_tumor_segmentation.h5')
IMG_SIZE = 128

# Load the trained model once
model = load_model(MODEL_PATH)

def predict_tumor_mask(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_mask = model.predict(img_array)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8).squeeze() * 255
    return pred_mask

def check_tumor(mask):
    return np.sum(mask) > 0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
                print(f"Created folder: {app.config['UPLOAD_FOLDER']}")

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved at: {filepath}, Exists: {os.path.exists(filepath)}")

            pred_mask = predict_tumor_mask(filepath)
            mask_filename = 'mask_' + filename
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
            cv2.imwrite(mask_path, pred_mask)
            print(f"Mask saved at: {mask_path}, Exists: {os.path.exists(mask_path)}")

            tumor_detected = check_tumor(pred_mask)

            return render_template('index.html',
                                   original_img=url_for('static', filename='uploads/' + filename),
                                   mask_img=url_for('static', filename='uploads/' + mask_filename),
                                   tumor=tumor_detected)

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
