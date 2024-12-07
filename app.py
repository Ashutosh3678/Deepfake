from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_file,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
from torch import nn
import shutil
from PIL import Image as pImage
import time
import glob



# -------------------------------  App  -----------------------------------------------

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "False"
db = SQLAlchemy(app)


# -------------------------------  Database  -----------------------------------------------


# Exampluuu tableuuu jyotiiii

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    



with app.app_context():
    db.create_all()
# -----------------------------   functions   -----------------------------------------------------

# -----------------------------   Routes   -----------------------------------------------------
app.secret_key = "your_secret_key"

# Configuration
UPLOAD_FOLDER = './uploaded_videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
app.config['UPLOAD_FOLDER'] = './uploaded_videos'


# Define constants
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im_size, im_size)), transforms.ToTensor(), transforms.Normalize(mean, std)])




class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(2048, 2048, 1)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.linear1(x_lstm[:, -1, :])

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def im_convert(tensor):
    """Convert tensor to an image."""
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image
def split_and_predict(model, video_path, sequence_length=100):
    # Initialize a list to store results
    predictions = []
    confidences = []

    # Get total number of frames in the video
    total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

    # Split video into sub-videos (each of 100 frames)
    num_chunks = total_frames // sequence_length + (1 if total_frames % sequence_length != 0 else 0)
    
    dataset = ValidationDataset([video_path], sequence_length=sequence_length, transform=train_transforms)

    for chunk_idx in range(num_chunks):
        # Make sure the chunk index is within the range of available data
        if chunk_idx < len(dataset):
            # Get a sub-video chunk
            chunk_frames = dataset[chunk_idx]

            # Perform prediction on this chunk
            prediction, confidence = predict(model, chunk_frames)

            # Store the result
            predictions.append(prediction)
            confidences.append(confidence)
        else:
            print(f"Warning: Skipping chunk {chunk_idx} as it exceeds available frames.")
    
    if predictions:
        # Average out the results
        avg_confidence = np.mean(confidences)
        final_prediction = np.argmax(np.bincount(predictions))  # Majority vote
    else:
        # Handle the case where no predictions are made
        avg_confidence = 0.0
        final_prediction = 0

    return final_prediction, avg_confidence



def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def predict(model, img):
    sm = nn.Softmax(dim=1)
    fmap, logits = model(img.to(device))
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return int(prediction.item()), confidence

# Routes
@app.route('/validate', methods=['GET', 'POST'])
def validate():
    if request.method == 'POST':
        video_file = request.files['video_file']
        sequence_length = int(request.form['sequence_length'])

        if not video_file or not allowed_video_file(video_file.filename):
            return render_template('index.html', error="Invalid video file")

        if video_file:
            video_file_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_file_path)

            session['file_name'] = video_file_path
            session['sequence_length'] = sequence_length

            return redirect(url_for('predict_page'))

    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    if 'file_name' not in session:
        return redirect(url_for('index'))

    video_file = session['file_name']
    sequence_length = session['sequence_length']
    dataset = ValidationDataset([video_file], sequence_length=sequence_length, transform=train_transforms)

    # Load model and perform prediction
    model = Model(2)
    model.load_state_dict(torch.load('path_to_model.pth', map_location=device))
    model.eval()

    result = predict(model, dataset[0])
    prediction = "REAL" if result[0] == 1 else "FAKE"
    confidence = round(result[1], 1)

    return render_template('predict.html', prediction=prediction, confidence=confidence)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/')
def home():
    if "user" in session:
        return render_template('index.html')
    else:
        return redirect(url_for("login"))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/profile')
def profile():
    if "user" in session:
        user=User.query.get(session["user"])
        return render_template('profile.html',user=user)
    else:
        return redirect(url_for("login"))



@app.route("/home/influencer/edit", methods=["POST"])
def edit_influencer_post():
            if "user" in session:
                user = User.query.get(session["user"])

            username = request.form.get("username")
            email = request.form.get("email")

        
            if email:
                user.email = email
            if username:
                user.username = username

            db.session.commit()

            flash("Profile updated successfully")
            return redirect(url_for("profile"))



@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/signup',methods=["POST"])
def signup_post():
    email = request.form.get("email")
    username = request.form.get("username")
    password1 = request.form.get("password1")
    password2 = request.form.get("password2")

    if not username or not email or not password1 or not password2:
            flash("Please enter your details")
            return redirect(url_for("signup_sponsor"))

    if password1 != password2:
            flash("Passwords do not match")
            return redirect(url_for("signup_sponsor"))

    user = User.query.filter_by(username=username).first()

    if user:
            flash("Username already exists")
            return redirect(url_for("signup_sponsor"))

    password_hash = generate_password_hash(password1)

    new_user = User(email=email, username=username, password=password_hash)
    db.session.add(new_user)
    db.session.commit()


    return redirect(url_for("login"))


@app.route("/login", methods=["POST"])
def login_post():
    print("request recived")
    username = request.form.get("username")
    password = request.form.get("password")

    if not username or not password:
        flash("Please fill out all fields")
        return redirect(url_for("login"))

    if username == "admin" and password == "admin":
        session["user"] = "admin"
        return redirect(url_for("admin"))

    user = User.query.filter_by(username=username).first()

    if not user:
        flash("Username does not exist")
        return redirect(url_for("login"))

    if not check_password_hash(user.password, password):
        flash("Incorrect password")
        return redirect(url_for("login"))

    session["user"] = user.id
    flash("Login successful")

    return redirect(url_for("profile"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("login")

# if __name__ == '__main__':
#     app.run(debug=True)