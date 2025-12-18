# app.py
from flask import Flask, render_template, request, redirect, url_for, session, g, flash, jsonify # Import jsonify
import joblib
import pandas as pd
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
from functools import wraps # For creating the login_required decorator

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here_12345!@#$' # Ensure this is a strong secret key

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'bmp', 'png', 'jpg', 'jpeg', 'gif'}
DATABASE = 'users.db'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Database Initialization and Connection ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        db.commit()
    print("Database initialized.")

init_db()

# --- Model and Encoders Loading ---
try:
    model = joblib.load('model/diabetes_model.pkl')
    le_fingerprint = joblib.load('model/le_fingerprint.pkl')
    le_family_history = joblib.load('model/le_family_history.pkl')
    le_diabetes_risk = joblib.load('model/le_diabetes_risk.pkl')
    print("Model and encoders loaded successfully.")
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    print("Please ensure 'training_script.py' has been run and generated the 'model' directory with all necessary files.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Fingerprint Feature Extraction Logic (No Change) ---
def predict_from_image(image_path):
    image = Image.open(image_path).convert("L")
    image_cv = np.array(image)
    image_cv = cv2.resize(image_cv, (300, 300))
    equalized = cv2.equalizeHist(image_cv)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    def thinning_iteration(img, iter):
        marker = np.zeros(img.shape, dtype=np.uint8)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                p2 = img[i-1, j]
                p3 = img[i-1, j+1]
                p4 = img[i, j+1]
                p5 = img[i+1, j+1]
                p6 = img[i+1, j]
                p7 = img[i+1, j-1]
                p8 = img[i, j-1]
                p9 = img[i-1, j-1]
                A = (p2 == 0 and p3 == 255) + (p3 == 0 and p4 == 255) + \
                    (p4 == 0 and p5 == 255) + (p5 == 0 and p6 == 255) + \
                    (p6 == 0 and p7 == 255) + (p7 == 0 and p8 == 255) + \
                    (p8 == 0 and p9 == 255) + (p9 == 0 and p2 == 255)
                B = (p2 == 255) + (p3 == 255) + (p4 == 255) + (p5 == 255) + \
                    (p6 == 255) + (p7 == 255) + (p8 == 255) + (p9 == 255)

                m1_condition_iter0 = (p2 == 0 or p4 == 0 or p6 == 0)
                m2_condition_iter0 = (p4 == 0 or p6 == 0 or p8 == 0)

                m1_condition_iter1 = (p2 == 0 or p4 == 0 or p8 == 0)
                m2_condition_iter1 = (p2 == 0 or p6 == 0 or p8 == 0)

                if iter == 0:
                    if A == 1 and 2 <= B <= 6 and m1_condition_iter0 and m2_condition_iter0:
                        marker[i, j] = 1
                else: # iter == 1
                    if A == 1 and 2 <= B <= 6 and m1_condition_iter1 and m2_condition_iter1:
                        marker[i, j] = 1
        return marker

    def thinning(img):
        img = img.copy()
        img[img > 0] = 255
        prev = np.zeros(img.shape, np.uint8)
        while True:
            marker = thinning_iteration(img, 0)
            img[marker == 1] = 0
            marker = thinning_iteration(img, 1)
            img[marker == 1] = 0
            diff = cv2.absdiff(img, prev)
            if cv2.countNonZero(diff) == 0:
                break
            prev = img.copy()
        return img

    thinned_image = thinning(inverted)
    ridge_count = cv2.countNonZero(thinned_image) // 300

    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_complexity = sum(len(cnt) for cnt in contours)

    if shape_complexity < 300:
        fingerprint_type = "arch"
    elif shape_complexity < 600:
        fingerprint_type = "loop"
    else:
        fingerprint_type = "whorl"

    return fingerprint_type, ridge_count

# --- Data Processing for Chart.js (No Change) ---
def get_chart_data():
    df = pd.read_csv('fingerprint_diabetes_dataset.csv')

    chart_data_package = {}

    RISK_COLORS = {
        'low': 'rgba(52, 211, 153, 0.8)', # green-400
        'medium': 'rgba(251, 191, 36, 0.8)', # amber-400
        'high': 'rgba(239, 68, 68, 0.8)' # red-500
    }
    BACKGROUND_COLORS = list(RISK_COLORS.values())
    BORDER_COLORS = [color.replace('0.8', '1') for color in BACKGROUND_COLORS]

    CATEGORY_PALETTE = [
        'rgba(99, 102, 241, 0.8)', # indigo-500
        'rgba(139, 92, 246, 0.8)', # purple-500
        'rgba(217, 70, 239, 0.8)', # fuchsia-500
        'rgba(236, 72, 153, 0.8)', # pink-500
        'rgba(244, 63, 94, 0.8)', # rose-500
        'rgba(251, 191, 36, 0.8)', # amber-400
        'rgba(163, 230, 53, 0.8)', # lime-400
        'rgba(34, 211, 238, 0.8)'  # cyan-400
    ]
    CATEGORY_BORDER_PALETTE = [color.replace('0.8', '1') for color in CATEGORY_PALETTE]

    # 1. Diabetes Risk Distribution (Pie Chart)
    risk_counts = df['Diabetes_Risk'].value_counts().sort_index()
    chart_data_package['diabetesRiskPieData'] = {
        'labels': risk_counts.index.tolist(),
        'datasets': [{
            'data': risk_counts.values.tolist(),
            'backgroundColor': [RISK_COLORS[label] for label in risk_counts.index],
            'borderColor': [color.replace('0.8', '1') for color in [RISK_COLORS[label] for label in risk_counts.index]],
            'borderWidth': 1
        }]
    }

    # 2. Fingerprint Type Distribution (Doughnut Chart)
    fp_type_counts = df['Fingerprint_Type'].value_counts()
    chart_data_package['fingerprintTypeDoughnutData'] = {
        'labels': fp_type_counts.index.tolist(),
        'datasets': [{
            'data': fp_type_counts.values.tolist(),
            'backgroundColor': CATEGORY_PALETTE[:len(fp_type_counts)],
            'borderColor': CATEGORY_BORDER_PALETTE[:len(fp_type_counts)],
            'borderWidth': 1
        }]
    }

    # 3. Family History Distribution (Pie Chart)
    fam_hist_counts = df['Family_History'].value_counts()
    chart_data_package['familyHistoryPieData'] = {
        'labels': fam_hist_counts.index.tolist(),
        'datasets': [{
            'data': fam_hist_counts.values.tolist(),
            'backgroundColor': [CATEGORY_PALETTE[0], CATEGORY_PALETTE[1]],
            'borderColor': [CATEGORY_BORDER_PALETTE[0], CATEGORY_BORDER_PALETTE[1]],
            'borderWidth': 1
        }]
    }

    # 4. Average Age by Diabetes Risk (Bar Chart)
    avg_age_by_risk = df.groupby('Diabetes_Risk')['Age'].mean().sort_index()
    chart_data_package['avgAgeBarData'] = {
        'labels': avg_age_by_risk.index.tolist(),
        'datasets': [{
            'label': 'Average Age',
            'data': avg_age_by_risk.values.tolist(),
            'backgroundColor': [RISK_COLORS[label] for label in avg_age_by_risk.index],
            'borderColor': [color.replace('0.8', '1') for color in [RISK_COLORS[label] for label in avg_age_by_risk.index]],
            'borderWidth': 1
        }]
    }

    # 5. Average BMI by Diabetes Risk (Bar Chart)
    avg_bmi_by_risk = df.groupby('Diabetes_Risk')['BMI'].mean().sort_index()
    chart_data_package['avgBmiBarData'] = {
        'labels': avg_bmi_by_risk.index.tolist(),
        'datasets': [{
            'label': 'Average BMI',
            'data': avg_bmi_by_risk.values.tolist(),
            'backgroundColor': [RISK_COLORS[label] for label in avg_bmi_by_risk.index],
            'borderColor': [color.replace('0.8', '1') for color in [RISK_COLORS[label] for label in avg_bmi_by_risk.index]],
            'borderWidth': 1
        }]
    }

    # 6. Fingerprint Type vs. Diabetes Risk (Stacked Bar Chart)
    fp_risk_crosstab = pd.crosstab(df['Fingerprint_Type'], df['Diabetes_Risk'])
    
    all_risk_levels = ['low', 'medium', 'high']
    for level in all_risk_levels:
        if level not in fp_risk_crosstab.columns:
            fp_risk_crosstab[level] = 0
    fp_risk_crosstab = fp_risk_crosstab[all_risk_levels].sort_index()

    datasets = []
    for risk_level in all_risk_levels:
        datasets.append({
            'label': risk_level.title(),
            'data': fp_risk_crosstab[risk_level].tolist(),
            'backgroundColor': RISK_COLORS[risk_level],
            'borderColor': RISK_COLORS[risk_level].replace('0.8', '1'),
            'borderWidth': 1
        })

    chart_data_package['fpTypeRiskStackedBarData'] = {
        'labels': fp_risk_crosstab.index.tolist(),
        'datasets': datasets
    }
    
    # 7. Ridge Count vs Diabetes Risk (Line Chart - Avg Ridge Count per Risk)
    avg_rc_by_risk = df.groupby('Diabetes_Risk')['Ridge_Count'].mean().sort_index()
    chart_data_package['ridgeCountLineData'] = {
        'labels': avg_rc_by_risk.index.tolist(),
        'datasets': [{
            'label': 'Average Ridge Count',
            'data': avg_rc_by_risk.values.tolist(),
            'borderColor': BACKGROUND_COLORS[0].replace('0.8', '1'),
            'backgroundColor': BACKGROUND_COLORS[0],
            'tension': 0.1,
            'fill': False
        }]
    }

    return chart_data_package

# --- Login Required Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'info')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes for New Pages ---

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/remidy')
def remidy():
    # Retrieve diabetes_risk from query parameters
    diabetes_risk = request.args.get('diabetes_risk', 'LOW').upper() # Default to LOW if not provided
    return render_template('remidy.html', diabetes_risk=diabetes_risk)

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        db = get_db()
        cursor = db.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            db.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login_page'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose a different one.', 'danger')
        except Exception as e:
            flash(f'An error occurred: {e}', 'danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        db = get_db()
        cursor = db.cursor()
        user = cursor.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username'] # Store username in session
            flash('Logged in successfully!', 'success')
            return redirect(url_for('upload_page')) # Redirect to the fingerprint upload page after login
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index_page'))

# --- Dashboard Page Route (No Change) ---
@app.route('/dashboard')
@login_required
def dashboard_page():
    chart_data = get_chart_data()
    return render_template('dashboard.html', chart_data=chart_data)


# --- Routes for Prediction Flow (MODIFIED for AJAX) ---

@app.route('/upload_fingerprint')
@login_required
def upload_page():
    # Reset session data for fingerprint if page is reloaded
    session.pop('fingerprint_type', None)
    session.pop('ridge_count', None)
    return render_template('upload_fingerprint.html', fingerprint_type=None, ridge_count=None, error=None)

@app.route('/extract_features', methods=['POST'])
@login_required
def extract_features_ajax(): # Renamed to avoid confusion with template rendering
    if 'fingerprint_image' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['fingerprint_image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath) # Save the file first
            fingerprint_type_str, ridge_count = predict_from_image(filepath)

            # Store in session for the next step of the prediction flow
            session['fingerprint_type'] = fingerprint_type_str
            session['ridge_count'] = ridge_count

            os.remove(filepath) # Clean up the uploaded file after processing

            return jsonify({
                'success': True,
                'fingerprint_type': fingerprint_type_str,
                'ridge_count': ridge_count
            })

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath) # Clean up if error occurred during processing
            print(f"Error during fingerprint processing: {e}") # Log the error
            return jsonify({'success': False, 'error': f"Error processing image: {e}"})
    else:
        return jsonify({'success': False, 'error': 'Allowed image types are bmp, png, jpg, jpeg, gif'})


@app.route('/prediction_form')
@login_required
def show_prediction_form():
    fingerprint_type = session.get('fingerprint_type')
    ridge_count = session.get('ridge_count')

    if fingerprint_type is None or ridge_count is None:
        flash("Fingerprint data missing. Please upload a fingerprint image first.", 'danger')
        return redirect(url_for('upload_page'))

    return render_template('prediction_form.html',
                           fingerprint_type=fingerprint_type,
                           ridge_count=ridge_count)

@app.route('/predict_diabetes', methods=['POST'])
@login_required
def predict_diabetes():
    fingerprint_type_str = request.form.get('fingerprint_type_hidden')
    ridge_count_str = request.form.get('ridge_count_hidden')

    if fingerprint_type_str is None or ridge_count_str is None:
        flash("Fingerprint data missing. Please re-upload.", 'danger')
        return redirect(url_for('upload_page'))

    try:
        ridge_count = int(ridge_count_str)
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        family_history_str = request.form['family_history']

        try:
            fingerprint_type_encoded = le_fingerprint.transform([fingerprint_type_str.lower()])[0]
        except ValueError:
            flash(f"Calculated Fingerprint Type: '{fingerprint_type_str}' is not in trained categories ('arch', 'loop', 'whorl').", 'danger')
            return render_template('prediction_form.html',
                                   fingerprint_type=fingerprint_type_str,
                                   ridge_count=ridge_count)

        try:
            family_history_encoded = le_family_history.transform([family_history_str.lower()])[0]
        except ValueError:
            flash(f"Invalid Family History: '{family_history_str}'. Please use 'yes' or 'no'.", 'danger')
            return render_template('prediction_form.html',
                                   fingerprint_type=fingerprint_type_str,
                                   ridge_count=ridge_count)

        input_data = pd.DataFrame([[fingerprint_type_encoded, ridge_count, age, bmi, family_history_encoded]],
                                  columns=['Fingerprint_Type', 'Ridge_Count', 'Age', 'BMI', 'Family_History'])

        prediction_encoded = model.predict(input_data)[0]
        diabetes_risk = le_diabetes_risk.inverse_transform([prediction_encoded])[0]

        return render_template('result.html',
                               fingerprint_type=fingerprint_type_str,
                               ridge_count=ridge_count,
                               age=age,
                               bmi=bmi,
                               family_history=family_history_str,
                               diabetes_risk=diabetes_risk.upper())

    except ValueError as ve:
        flash(f"Input Error: {ve}. Please check your numerical values.", 'danger')
        return render_template('prediction_form.html',
                               fingerprint_type=fingerprint_type_str,
                               ridge_count=ridge_count)
    except KeyError as ke:
        flash(f"Missing form data: {ke}. Please fill all fields.", 'danger')
        return render_template('prediction_form.html',
                               fingerprint_type=fingerprint_type_str,
                               ridge_count=ridge_count)
    except Exception as e:
        flash(f"An unexpected error occurred during prediction: {e}", 'danger')
        return render_template('prediction_form.html',
                               fingerprint_type=fingerprint_type_str,
                               ridge_count=ridge_count)


if __name__ == '__main__':
    app.run(debug=True)