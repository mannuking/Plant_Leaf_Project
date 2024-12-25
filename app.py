import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import openai
from werkzeug.utils import secure_filename
import torch.nn as nn

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config['MODEL_PATH'] = 'models/custom_vision_model.pth'
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    print("Error: OPENAI_API_KEY environment variable not set.")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Model Definition (Custom CNN)
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 14 * 14, 256)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = image_transforms(image).unsqueeze(0)
    return tensor

def detect_anomalies(image_tensor, vision_model):
    with torch.no_grad():
        output = vision_model(image_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

def generate_explanation(label):
    prompt = f"The plant leaf shows signs of {label}. Provide possible causes, prevention, and treatment options."
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": "You are a helpful assistant that provides information about plant leaf diseases."},
            {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Open AI error: {e}")
        return "Error: Failed to generate explanation with OpenAI API."

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def compute_and_highlight_ssim(image1_path, image2_path):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    score, diff = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    highlighted_image = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return score, highlighted_image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare_images():
    if "image1" not in request.files or "image2" not in request.files:
        return redirect(request.url)
    image1 = request.files["image1"]
    image2 = request.files["image2"]
    if image1.filename == "" or image2.filename == "":
        return "Please select both images!"
    if image1 and allowed_file(image1.filename) and image2 and allowed_file(image2.filename):
        image1_filename = secure_filename(image1.filename)
        image2_filename = secure_filename(image2.filename)
        image1_path = os.path.join(app.config["UPLOAD_FOLDER"], image1_filename)
        image2_path = os.path.join(app.config["UPLOAD_FOLDER"], image2_filename)
        image1.save(image1_path)
        image2.save(image2_path)
        ssim_score, highlighted_image = compute_and_highlight_ssim(image1_path, image2_path)
        highlighted_filename = "highlighted_" + image2_filename
        highlighted_path = os.path.join(app.config["UPLOAD_FOLDER"], highlighted_filename)
        cv2.imwrite(highlighted_path, highlighted_image)
        return render_template(
            "result.html",
            ssim_score=f"{ssim_score:.4f}",
            image1=image1_filename,
            image2=image2_filename,
            highlighted_image=highlighted_filename,
        )
    else:
        return "Invalid file type. Allowed types are png, jpg, jpeg."

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    image_tensor = process_image(file_path)
    if check_and_load_model():
       predicted_index = detect_anomalies(image_tensor, app.vision_model) # get the index of the label
       labels = ["Apple___healthy", "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Background_without_leaves", "Blueberry___healthy",
                  "Cherry___healthy","Cherry___Powdery_mildew", "Corn___Cercospora_leaf_spot Gray_leaf_s...", "Corn___Common_rust", "Corn___healthy",
                  "Corn___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                  "Orange___Haunglongbing_(Citrus_greeni...", "Peach___Bacterial_spot", "Peach___healthy", "Pepper_bell___Bacterial_spot", "Pepper_bell___healthy",
                   "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
                  "Strawberry___healthy","Strawberry___Leaf_scorch", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
                  "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spi...", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus",
                  "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]
       anomaly_label = labels[predicted_index] # Get the string label from the predicted index.
       explanation = generate_explanation(anomaly_label) # Use the predicted anomaly to generate the text
       return render_template(
            "result.html",
            anomaly_label=anomaly_label,
            explanation=explanation,
        )
    else:
        return jsonify({'error' : "Model not trained. Run train.py first."}), 500

def check_and_load_model():
    if not hasattr(app, 'vision_model_loaded') or not app.vision_model_loaded:
        if os.path.exists(app.config['MODEL_PATH']):
            print("Loading vision model from saved path...")
            num_classes = 39  # Number of classes in the trained model
            vision_model = CustomCNN(num_classes = num_classes)  #Load custom model instead of resnet
            vision_model.load_state_dict(torch.load(app.config['MODEL_PATH'],map_location=torch.device('cpu')))
            vision_model.eval()
            app.vision_model = vision_model
            app.vision_model_loaded = True
        else:
            print("Trained model not found. Run train.py first.")
            return False
    return True

@app.route('/test_results', methods=['GET'])
def test_results():
    if check_and_load_model():
         image_path1 = 'static/uploads/known1.jpg'
         image_path2 = 'static/uploads/unknown1.jpg'
         try:
            image_tensor1 = process_image(image_path1)
            predicted_index = detect_anomalies(image_tensor1, app.vision_model)
            labels = ["Apple___healthy", "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Background_without_leaves", "Blueberry___healthy",
                  "Cherry___healthy","Cherry___Powdery_mildew", "Corn___Cercospora_leaf_spot Gray_leaf_s...", "Corn___Common_rust", "Corn___healthy",
                  "Corn___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                  "Orange___Haunglongbing_(Citrus_greeni...", "Peach___Bacterial_spot", "Peach___healthy", "Pepper_bell___Bacterial_spot", "Pepper_bell___healthy",
                   "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
                  "Strawberry___healthy","Strawberry___Leaf_scorch", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
                  "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spi...", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus",
                  "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]
            anomaly_label = labels[predicted_index]
            explanation = generate_explanation(anomaly_label)
            ssim_score, highlighted_image = compute_and_highlight_ssim(image_path1, image_path2)

            highlighted_filename = "highlighted_test.jpg"
            highlighted_path = os.path.join(app.config["UPLOAD_FOLDER"], highlighted_filename)
            cv2.imwrite(highlighted_path, highlighted_image)
            return render_template(
                "result.html",
                ssim_score=f"{ssim_score:.4f}",
                image1='known1.jpg',
                image2='unknown1.jpg',
                highlighted_image=highlighted_filename,
                anomaly_label=anomaly_label,
                explanation=explanation
            )
         except Exception as e:
            return jsonify({'error' : str(e)}), 500
    else:
       return jsonify({'error' : "Model not trained. Run train.py first."}), 500

if __name__ == "__main__":
    app.run(debug=True)
