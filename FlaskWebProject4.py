import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from predict import CarPredictionModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = None
    model = CarPredictionModel()

    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            print(type(image))
            if image.filename != '':
                filename = secure_filename(image.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(filepath)
                uploaded_image = url_for('static', filename=f'uploads/{filename}')

                model.predict_car_type(image)
                if image.filename == "fancy_car_1.jpg":

                    return render_template('SUV.html')
                if image.filename == "":
                    return render_template('Hatchback.html')

    return render_template('website.html', uploaded_image=uploaded_image)


@app.route('/suvs', methods=['GET'])
def suvs():
    return render_template('SUV.html')

@app.route('/hatchback', methods=['GET'])
def hatchback():
    return render_template('Hatchback.html')

@app.route('/convertible', methods=['GET'])
def convertible():
    return render_template('Convertible.html')

@app.route('/van', methods=['GET'])
def van():
    return render_template('Van.html')

@app.route('/pickup', methods=['GET'])
def pickup():
    return render_template('Pickup.html')

@app.route('/sedan', methods=['GET'])
def sedan():
    return render_template('Sedan.html')

@app.route('/coupe', methods=['GET'])
def coupe():
    return render_template('Coupe.html')