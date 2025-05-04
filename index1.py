import os
import re
import cv2
import pickle
import numpy as np
import tensorflow as tf
import pytesseract
import easyocr
import threading
from ultralytics import YOLO
from tkinter import Tk, Label, Button, filedialog, Canvas, Frame
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.cluster import KMeans
import pyttsx3
import speech_recognition as sr #Import speech recognition

from train_model import extract_serial_number
from train_model import extract_total_features


# Configure Tesseract OCR
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def capture_from_webcam():
    global image_path  # Important: Use your existing variable

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("📸 Webcam - Press Space to Capture", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow("📸 Webcam - Press Space to Capture", frame)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC key to exit
            print("❌ Escape hit, closing...")
            break

        elif key % 256 == 32:  # SPACE key to capture
            captured_image = "captured_webcam_image.jpg"
            cv2.imwrite(captured_image, frame)
            print(f"✅ Image captured and saved as {captured_image}")

            image_path = captured_image  # ✅ Update your existing image_path variable

            cap.release()
            cv2.destroyAllWindows()

            # ✅ Directly call update_results with image_path
            update_results(image_path)
            break

    cap.release()
    cv2.destroyAllWindows()

# Import training model module
import train_model  # Ensure train_model.py is in the same directory

# Load pre-trained models
try:
    print("🔄 Loading trained models...")
    
    if os.path.exists("currency_classifier_model.keras"):
        model = load_model("currency_classifier_model.keras")  # CNN Classification Model
    else:
        print("⚠️ CNN model not found. Please train the model first.")
        exit()
    
    ml_models = train_model.load_ml_models()  # Load KNN, RF
    if ml_models is None:
        print("⚠️ ML models not found. Please train them first.")
        exit()
    
    cnn_models = train_model.load_cnn_model() #cnn model load
    if cnn_models is None:
        print("⚠️ ML models not found. Please train them first.")
        exit()

    yolo_model = YOLO("yolov8n.pt")  # YOLO Object Detection
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# Initialize EasyOCR for text recognition
reader = easyocr.Reader(['en', 'hi', 'mr'])

# Class labels
class_labels = ['Fake', 'Real']

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Initialize Speech Recognition
recognizer = sr.Recognizer()

import pyttsx3

engine = pyttsx3.init(driverName='sapi5')  # Windows friendly

def speak(text):
    try:
        engine.endLoop()
    except Exception:
        pass
    engine.say(text)
    engine.runAndWait()

# Function to listen for voice commands
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""

# ✅ Function to process & display feature images (Right-Side Alignment)
def process_and_display_images(image_path):
    """
    Calls `train_model.extract_and_save_feature_images()` and updates GUI.
    """
    train_model.extract_and_save_feature_images(image_path)  # Process and save images

    # ✅ Load processed images
    gray_img = Image.open("processed_gray.jpg").resize((350, 350), Image.LANCZOS)
    edges_img = Image.open("processed_edges.jpg").resize((350, 350), Image.LANCZOS)
    cnn_img = Image.open("processed_cnn.jpg").resize((350, 350), Image.LANCZOS)

    # ✅ Convert images to Tkinter format
    gray_tk = ImageTk.PhotoImage(gray_img)
    edges_tk = ImageTk.PhotoImage(edges_img)
    cnn_tk = ImageTk.PhotoImage(cnn_img)

    # ✅ Display images on the right side of the GUI
    gray_canvas.create_image(75, 75, image=gray_tk)
    gray_canvas.image = gray_tk

    edges_canvas.create_image(75, 75, image=edges_tk)
    edges_canvas.image = edges_tk

    cnn_canvas.create_image(75, 75, image=cnn_tk)
    cnn_canvas.image = cnn_tk

# Function to detect objects on the currency note
def detect_objects(image_path):
    results = yolo_model(image_path)
    return results[0].boxes.data.cpu().numpy()

# Function to highlight features
def highlight_features(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ✅ YOLO Detection
    results = yolo_model(image_path)
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = map(int, box)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for detected objects

    # ✅ Contour Detection (Edge-based)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500:
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for feature boxes

    # ✅ Save extracted feature image
    extracted_feature_path = "extracted_features.jpg"
    cv2.imwrite(extracted_feature_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return extracted_feature_path
#Function to call images
def update_grayscale_image(image_path):
    train_model.save_grayscale_image(image_path)
    img = Image.open("processed_gray.jpg").resize((350, 150), Image.LANCZOS)  # Resize to 350x150
    gray_tk = ImageTk.PhotoImage(img)
    gray_canvas.create_image(175, 75, image=gray_tk)  # Centered: half of width and height
    gray_canvas.image = gray_tk

def update_edge_image(image_path):
    train_model.save_edge_image(image_path)
    img = Image.open("processed_edges.jpg").resize((350, 150), Image.LANCZOS)  # Resize
    edges_tk = ImageTk.PhotoImage(img)
    edges_canvas.create_image(175, 75, image=edges_tk)  # Centered
    edges_canvas.image = edges_tk

def update_cnn_preprocessed_image(image_path):
    train_model.save_cnn_preprocessed_image(image_path)
    img = Image.open("processed_cnn.jpg").resize((350, 150), Image.LANCZOS)  # Resize
    cnn_tk = ImageTk.PhotoImage(img)
    cnn_canvas.create_image(175, 75, image=cnn_tk)  # Centered
    cnn_canvas.image = cnn_tk


# Function to update GUI results
def update_results(image_path):
    def process():
        result = train_model.predict_with_models(image_path)
        train_model.log_scan_history_text(image_path, result)

        # ✅ Load and display uploaded image
        img = Image.open(image_path).resize((250, 250), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        main_canvas.create_image(125, 125, image=img_tk)
        main_canvas.image = img_tk

        # ✅ Extract and show total features
        total_features = extract_total_features(image_path)

        # ✅ Highlight and display feature-detected image
        feature_img_path = highlight_features(image_path)
        feature_img = Image.open(feature_img_path).resize((250, 250), Image.LANCZOS)
        feature_tk = ImageTk.PhotoImage(feature_img)
        feature_canvas.create_image(125, 125, image=feature_tk)
        feature_canvas.image = feature_tk

        # ✅ Feature check results
        f = result['feature_checks']
        feature_summary = (
            f"Ashoka Pillar: {'✅' if f['ashoka_pillar'] else '❌'}\n"
            f"RBI Logo: {'✅' if f['rbi_logo'] else '❌'}\n"
            f"RBI Text (Hindi/English): {'✅' if f['rbi_text'] else '❌'}\n"
            f"Gandhi Image: {'✅' if f['gandhi_image'] else '❌'}\n"
            f"Governor Signature: {'✅' if f['governor_signature'] else '❌'}\n"
            f"Denomination Code: {'✅' if f['denomination_code'] else '❌'}\n"
        )

        # ✅ Result message
        result_text = (
            f"Status: {result['status']}\n"
            f"Serial Number: {result['serial']}\n"
            f"ML Prediction: {result['ml_prediction']} ({result['ml_confidence']:.2f}%)\n"
            f"CNN Prediction: {result['cnn_prediction']} ({result['cnn_confidence']:.2f}%)\n"
            f"Bleed Lines Detected: {result['bleed_lines']}\n"
            f"Detected Note Type: ₹{result['note_type']}\n"
            f"🔍 Feature Verification:\n{result['feature_summary']}"
        )

        result_label.config(text=result_text, fg="green" if result['status'] == "Real" else "red")
        speak(result_text)
        speak("please again select image & scan to  upload the image.")
        speak("Select Feature of Currecny to see total important feature of each denomination")
        speak("Select Ml confusion matrix or CNN confusion matrix to see train model confusion matrixs")
        speak("Select Confidence Graph to see uploaded image feature matching confidence")
    threading.Thread(target=process).start()

#Hishtory
def show_scan_history():
    import tkinter as tk
    from tkinter import Toplevel, Text, Scrollbar, RIGHT, Y, LEFT, BOTH, messagebox

    history_file = "scan_history.txt"

    if not os.path.exists(history_file):
        messagebox.showinfo("History", "No history found yet.")
        return

    history_window = Toplevel()
    history_window.title("Scan History")
    history_window.geometry("800x400")
    history_window.configure(bg='#000000')  # 🖤 Black background

    text_widget = Text(
        history_window,
        wrap='none',
        font=("Courier", 10),
        bg='#000000',       # Background: black
        fg='#00FF00',       # Text: green
        insertbackground='#FFFFFF'  # Cursor color: white
    )
    text_widget.pack(side=LEFT, fill=BOTH, expand=True)

    scrollbar = Scrollbar(history_window, command=text_widget.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    text_widget.config(yscrollcommand=scrollbar.set)

    with open(history_file, 'r') as f:
        text_widget.insert('1.0', f.read())

    text_widget.config(state='disabled')  # Make it read-only



# Function to browse and select an image
last_image_path = ""

def browse_and_scan():
    global last_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        last_image_path = file_path
        update_results(file_path)


# Function for features file
import webbrowser
import tkinter as tk
def open_feature_pdf():
    webbrowser.open(r'file:///C:/Users/Apurva%20Gole/Documents/Data/Data/DATASET4/Feature.pdf')

def call_analyze_feature_confidences():
    if last_image_path:
        train_model.analyze_and_plot_feature_confidences(last_image_path)
    else:
        print("⚠️ No image selected yet.")


# GUI Setup
root = Tk()
root.title("Fake Currency Detection")
root.geometry("1800x600")
root.configure(bg="#000000")

# ✅ Main Frame (Left: Main Image, Right: Processed Images)
main_frame = Frame(root, bg="#000000")
main_frame.pack(pady=10, padx=10, fill="both", expand=True)

# ✅ Left Side (Main Image)
left_frame = Frame(main_frame, bg="#000000")
left_frame.pack(side="left", padx=20, pady=10)
# ✅ Middle Section (Extracted Features)
middle_frame = Frame(main_frame, bg="#000000")
middle_frame.pack(side="left", padx=20, pady=10)

Label(middle_frame, text="💰 Fake Currency Detector 💰", font=("Arial", 18, "bold"), fg="white", bg="#000000").pack(pady=10)

Label(left_frame, text="🔎 Highlighted Features", font=("Arial", 12), fg="white", bg="#000000").pack(pady=5)
feature_canvas = Canvas(left_frame, width=250, height=250, bg="gray", highlightthickness=2)
feature_canvas.pack(pady=5)

# ✅ Main Image Display
main_canvas = Canvas(left_frame, width=250, height=250, bg="gray", highlightthickness=2)
main_canvas.pack(pady=10)

Button(middle_frame, text="📁 Select Image & Scan", command=browse_and_scan, font=("Arial", 12, "bold"), bg="#007BFF", fg="white").pack(pady=10)
Button(middle_frame, text="Features of currency ", command=open_feature_pdf, font=("Arial", 12, "bold"), bg="#007BFF", fg="white").pack(pady=10)
Button(middle_frame, text="📊 ML Confusion Matrix", command=train_model.generate_ml_confusion_matrix, font=("Arial", 12, "bold"), bg="#6f42c1", fg="white").pack(pady=5)
Button(middle_frame, text="📊 CNN Confusion Matrix", command=train_model.generate_cnn_confusion_matrix, font=("Arial", 12, "bold"), bg="#17a2b8", fg="white").pack(pady=5)
Button(middle_frame, text="📊 Confidence Graph", command=call_analyze_feature_confidences, font=("Arial", 12, "bold"), bg="#48D1CC", fg="white").pack(pady=5)
Button(middle_frame, text="📸 Capture from Webcam", command=capture_from_webcam, font=("Arial", 12, "bold"), bg="#28a745", fg="white", width=25).pack(pady=10)
Button(middle_frame, text="📜 View Scan History", command=show_scan_history, font=("Arial", 12, "bold"), bg="#28a745", fg="white", width=25).pack(pady=10)

# ✅ Right Side (Processed Images)
right_frame = Frame(main_frame, bg="#000000")
right_frame.pack(side="right", padx=20, pady=10)

Label(middle_frame, text="📸 Processed Images", font=("Arial", 14, "bold"), fg="white", bg="#000000").pack(pady=5)

# Grayscale Button & Canvas
Button(right_frame, text="📁 Grayscale", command=lambda: update_grayscale_image(last_image_path), 
       font=("Arial", 12, "bold"), bg="#007BFF", fg="white").pack(pady=10)
Label(right_frame, text="Grayscale", font=("Arial", 10), fg="white", bg="#000000").pack(pady=5)
gray_canvas = Canvas(right_frame, width=350, height=150, bg="gray", highlightthickness=2)
gray_canvas.pack()

# Edge Detection Button & Canvas
Button(right_frame, text="📁 Edge Detection", command=lambda: update_edge_image(last_image_path), 
       font=("Arial", 12, "bold"), bg="#007BFF", fg="white").pack(pady=10)
Label(right_frame, text="Edge Detection", font=("Arial", 10), fg="white", bg="#000000").pack(pady=5)
edges_canvas = Canvas(right_frame, width=350, height=150, bg="gray", highlightthickness=2)
edges_canvas.pack()

# CNN Preprocessed Button & Canvas
Button(right_frame, text="📁 CNN Preprocessed", command=lambda: update_cnn_preprocessed_image(last_image_path), 
       font=("Arial", 12, "bold"), bg="#007BFF", fg="white").pack(pady=10)
Label(right_frame, text="CNN Preprocessed", font=("Arial", 10), fg="white", bg="#000000").pack(pady=5)
cnn_canvas = Canvas(right_frame, width=350, height=150, bg="gray", highlightthickness=2)
cnn_canvas.pack()




# ✅ Result Label
result_label = Label(middle_frame, text="📊 Prediction Results Here", font=("Arial", 14), fg="white", bg="#000000", wraplength=500, justify="center")
result_label.pack(pady=20)

speak("Hello, please select image & scan to  upload the image.")
root.mainloop()
