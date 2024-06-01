from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import load_model
from keras.src.applications.densenet import preprocess_input
from skimage.io import imread

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carregue seu modelo aqui
model = load_model('modelo_raca_caninas.h5')

breed_list = os.listdir("Images/")

# Mapeamento de rótulos
label_maps_rev = {}
for i, v in enumerate(breed_list):
    label_maps_rev.update({i : v})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = previsao_foto(filepath)
        return render_template('index.html', result=result)
    return redirect(request.url)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpeg', 'jpg'}

def previsao_foto(filepath):
    img = Image.open(filepath)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img.save(filepath)
    img = imread(filepath)
    img = preprocess_input(img)
    probs = model.predict(np.expand_dims(img, axis=0))
    predictions = []
    for idx in np.argsort(probs[0])[::-1][:5]:
        try:
            label = label_maps_rev[idx]
            predictions.append((f"{probs[0][idx]*100:.2f}%", label))
        except KeyError:
            predictions.append((f"{probs[0][idx]*100:.2f}%", f"Label {idx} não mapeado"))
    return {'image_url': url_for('static', filename=filepath), 'predictions': predictions}

if __name__ == '__main__':
    app.run(debug=True)
