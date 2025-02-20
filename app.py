from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'secret_key'  # Kunci rahasia untuk sesi
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/flask_app'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Route untuk redirect ke halaman login jika pengguna tidak terautentikasi

# Load model
# Load modelsx
model1 = load_model('model/ConvMixer_Final.h5')
model2 = load_model('model/ResNet_FinalisSHED.h5')  # Load the second model


# Class labels
class_labels = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis',
                'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis', 'unknown']

# Model Tabel untuk Riwayat Prediksi
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.id} - {self.predicted_class}>"

# Model Tabel untuk Pengguna
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f"<User {self.username} - {self.role}>"

    
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    contact = db.Column(db.String(50), nullable=False)
    address = db.Column(db.Text, nullable=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction.id'), nullable=True)

    def __repr__(self):
        return f"<Patient {self.name}>"

    
    # Penjelasan penyakit
disease_info = {
    'dyed-lifted-polyps': 'Polip yang diangkat dan diwarnai untuk pemeriksaan lebih lanjut.',
    'dyed-resection-margins': 'Area batas reseksi yang diwarnai setelah pengangkatan jaringan.',
    'esophagitis': 'Peradangan pada kerongkongan yang dapat disebabkan oleh refluks asam atau infeksi.',
    'normal-cecum': 'Cekum tampak normal, bagian pertama dari usus besar.',
    'normal-pylorus': 'Pilorus normal, bagian lambung yang menghubungkan ke usus kecil.',
    'normal-z-line': 'Garis Z normal, area transisi antara esofagus dan lambung.',
    'polyps': 'Pertumbuhan jaringan abnormal di dinding saluran pencernaan.',
    'ulcerative-colitis': 'Penyakit radang usus kronis yang menyebabkan luka di usus besar.',
    'unknown': 'Kondisi tidak teridentifikasi. Memerlukan konsultasi lebih lanjut atau gambar tidak valid'
}

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Halaman login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Verifikasi pengguna
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)  # Menggunakan Flask-Login untuk login
            flash('Login berhasil!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Username atau password salah!', 'danger')
    return render_template('login.html')

# Halaman utama
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# Halaman Prediksi
@app.route('/detect')
@login_required
def detect():
    return render_template('detect.html')

@app.route('/history')
@login_required
def history():
    # Ambil semua data prediksi dan pasien dari database
    predictions = db.session.query(Prediction, Patient).join(
        Patient, Prediction.id == Patient.prediction_id, isouter=True
    ).order_by(Prediction.timestamp.desc()).all()

    return render_template('history.html', predictions=predictions)


# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    # Validasi input file
    if 'file' not in request.files:
        flash('Tidak ada file yang diunggah!', 'danger')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('Nama file kosong!', 'danger')
        return redirect(request.url)

    # Ambil data pasien dari form
    name = request.form['name']
    age = int(request.form['age'])
    gender = request.form['gender']
    contact = request.form['contact']
    address = request.form.get('address', '')

    # Simpan data pasien ke database
    new_patient = Patient(name=name, age=age, gender=gender, contact=contact, address=address)
    db.session.add(new_patient)
    db.session.commit()

    # Pilih model untuk prediksi
    model_choice = request.form.get('model')

    if file:
        # Simpan file
        image_dir = 'static/images'
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        filepath = os.path.join(image_dir, file.filename)
        file.save(filepath)

        # Preprocessing gambar
        image = load_img(filepath, target_size=(224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Prediksi
        if model_choice == 'model1':
            predictions = model1.predict(image)
        elif model_choice == 'model2':
            predictions = model2.predict(image)
        
        predicted_class = class_labels[np.argmax(predictions)]
        predicted_prob = np.max(predictions)
        disease_description = disease_info.get(predicted_class, 'Penjelasan tidak tersedia.')

        # Simpan hasil prediksi ke database
        new_prediction = Prediction(image_path=filepath, predicted_class=predicted_class)
        db.session.add(new_prediction)
        db.session.commit()

        # Update pasien dengan ID prediksi
        new_patient.prediction_id = new_prediction.id
        db.session.commit()

        return render_template(
    'result.html',
    filename=file.filename,
    predicted_class=predicted_class,
    disease_description=disease_description,
    predicted_prob=predicted_prob,
    patient={
        'name': name,
        'age': age,
        'gender': gender,
        'contact': contact,
        'address': address
    }  # Kirim data pasien ke template
)

# Tentang Aplikasi
@app.route('/about')
@login_required
def about():
    return render_template('about.html')

# Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()  # Menggunakan Flask-Login untuk logout
    flash('Logout berhasil!', 'success')
    return redirect(url_for('login'))

# Menjalankan server
if __name__ == '__main__':
    # Membuat database jika belum ada
    with app.app_context():
        db.create_all()
    app.run(debug=True)
