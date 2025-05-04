import os
import pickle
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50, ResNet101, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from skimage.feature import local_binary_pattern, hog
from skimage import color
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import pytesseract
import re
import easyocr  # Import easyocr
import logging
import os
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


# Path Configuration
DATASET_PATH = r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4"
REAL_FEATURES_PATH = r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\Real Notes"
FAKE_FEATURES_PATH = r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\Fake Notes"


# ✅ Preprocess image function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224))
    gray = gray.astype(np.float32) / 255.0  # Normalize
    return gray

# ✅ Define ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='validation'
)

# ✅ Serial Number Extraction using OCR
# ✅ Serial Number Extraction using OCR (Updated)
# Initialize EasyOCR reader (at the module level)
reader = easyocr.Reader(['en'])  # 'en' for English language

def extract_serial_number(image_path, debug=True):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result = reader.readtext(gray)

    extracted_text = " ".join([detection[1] for detection in result])

    print(f"EasyOCR Extracted Text: {extracted_text}")

    extracted_text = re.sub(r'[^\w\s]', '', extracted_text)
    extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()

    words = extracted_text.split()

    serial_prefix = None
    serial_suffix = None
    for word in words:
        print(f"Checking word: {word}")
        if len(word) == 3: # Added length check here
            prefix_match = re.search(r'[0-9]{1}[A-Z]{2}', word)

            if prefix_match and not serial_prefix:
                serial_prefix = prefix_match.group(0)
                print(f"Prefix Match: {serial_prefix}")

        suffix_match = re.search(r'[0-9]{6}', word)

        if suffix_match and not serial_suffix:
            serial_suffix = suffix_match.group(0)
            print(f"Suffix Match: {serial_suffix}")
        
        if serial_prefix and serial_suffix:
            extracted_serial = f"{serial_prefix} {serial_suffix}"
            print(f"Full Serial Match: {extracted_serial}")
            return extracted_serial + ","

    # Check for prefix and suffix in entire string if not found in words
    if not serial_prefix:
        prefix_match = re.search(r'[0-9]{1}[A-Z]{2}', extracted_text)
        if prefix_match:
            serial_prefix = prefix_match.group(0)
            print(f"Prefix Match (entire text): {serial_prefix}")

    if not serial_suffix:
        suffix_match = re.search(r'[0-9]{6}', extracted_text)
        if suffix_match:
            serial_suffix = suffix_match.group(0)
            print(f"Suffix Match (entire text): {serial_suffix}")
    
    if serial_prefix and serial_suffix:
        extracted_serial = f"{serial_prefix} {serial_suffix}"
        print(f"Full Serial Match (entire text): {extracted_serial}")
        return extracted_serial + "Real,"
        

    print("Regex Match: None")
    return "Fake Serial Number"

def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")
    lbp_resized = cv2.resize(lbp, (64, 64)).flatten()
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
    features = np.hstack([lbp_resized, hog_features])
    return np.pad(features, (0, 5000 - len(features)), mode='constant') if len(features) < 5000 else features[:5000]



import cv2
import numpy as np

def detect_bleed_lines(image_path, show_output=False, expected_lines=5):
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        return False, 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # ROIs: left and right 10% regions
    left_roi = gray[:, :int(0.1 * width)]
    right_roi = gray[:, int(0.9 * width):]

    def preprocess_roi(roi):
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    left_edges = preprocess_roi(left_roi)
    right_edges = preprocess_roi(right_roi)

    def get_vertical_line_x_positions(edge_img):
        lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, threshold=50,
                                minLineLength=30, maxLineGap=5)
        x_positions = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 5 and abs(y2 - y1) > 20:
                    x_positions.append(x1)
        return x_positions

    left_xs = get_vertical_line_x_positions(left_edges)
    right_xs = [x + int(0.9 * width) for x in get_vertical_line_x_positions(right_edges)]

    all_xs = left_xs + right_xs

    # ⬇️ Increase min_spacing to debounce multiple lines better
    def cluster_x_positions(x_positions, min_spacing=18):
        if not x_positions:
            return []
        x_positions.sort()
        clusters = [x_positions[0]]
        for x in x_positions[1:]:
            if abs(x - clusters[-1]) >= min_spacing:
                clusters.append(x)
        return clusters

    unique_lines = cluster_x_positions(all_xs)
    bleed_count = len(unique_lines)

    # ±1 tolerance
    is_real = expected_lines - 1 <= bleed_count <= expected_lines + 1

    if show_output:
        vis_img = img.copy()
        for x in unique_lines:
            cv2.line(vis_img, (x, 0), (x, height), (0, 255, 0), 2)
        cv2.imshow('Detected Bleed Lines', vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return is_real, bleed_count


def get_note_type_by_bleed_lines(bleed_count):
    """
    Returns the denomination based on the number of bleed lines detected.
    Adjust values as needed based on real-world testing.
    
    Args:
        bleed_count (int): Number of detected bleed lines.
    
    Returns:
        str: Note denomination.
    """
    if bleed_count >= 5:
        return "500"
    elif bleed_count == 4:
        return "100"
    elif bleed_count == 2 or bleed_count == 3:
        return "200"
    else:
        return "10_20_50"

#function deblurga
def apply_deblurgan(img):
    # Load DeblurGAN2 (PyTorch)
    # Placeholder until integrated
    print("🔧 DeblurGAN placeholder applied.")
    return img

def preprocess_for_template_matching(img, use_esrgan=False, use_dncnn=False, use_deblurgan=False):
    img = apply_deblurgan(img)

    # Classical Enhancements
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray

#Function Ashoka pillar
def detect_ashoka_pillar(
    image_path,
    template_paths=[
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\ashoka_pillar_1.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\ashoka_pillar_2.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\ashoka_pillar_3.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\ashoka_pillar_4.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\ashoka_pillar_5.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\ashoka_pillar_6.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\ashoka_pillar_7.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\ashoka_pillar_8.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\ashoka_pillar_9.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\ashoka_pillar_10.jpg"
    ],
    yolo_model_path=r"C:\Users\Apurva Gole\Documents\Data\Data\yolov8n.pt",
    debug=False
):
    import cv2
    import numpy as np
    import os
    from ultralytics import YOLO

    note_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if note_img is None:
        print("❌ Error loading note image.")
        return "Fake Ashoka Pillar"

    # Enhance image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    note_img = cv2.filter2D(note_img, -1, sharpen_kernel)
    note_img = cv2.equalizeHist(note_img)
    note_img = cv2.fastNlMeansDenoising(note_img, h=10)

    orb = cv2.ORB_create(nfeatures=1000)
    kp_note, des_note = orb.detectAndCompute(note_img, None)

    if des_note is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for template_path in template_paths:
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                print(f"❌ Could not load template: {template_path}")
                continue

            kp_temp, des_temp = orb.detectAndCompute(template_img, None)
            if des_temp is None:
                continue

            matches = bf.match(des_temp, des_note)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]
            confidence = min(len(good_matches) * 10, 100)

            if debug:
                print(f"🔍 Matching {template_path} → Good Matches: {len(good_matches)}")
                if len(kp_temp) > 0 and len(kp_note) > 0:
                    match_img = cv2.drawMatches(template_img, kp_temp, note_img, kp_note, matches[:20], None, flags=2)
                    cv2.imshow(f"Match with {template_path}", match_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            if len(good_matches) >= 10:
                print(f"✅ Ashoka Pillar detected using {template_path}")
                return "Real,", confidence

    print("🔄 ORB matching failed — trying YOLO Ashoka Pillar detection...")

    try:
        yolo_model = YOLO(yolo_model_path)
        results = yolo_model(image_path)
        boxes = results[0].boxes

        if debug:
            print(f"📦 YOLO detected {len(boxes)} objects.")
            #results[0].show()

        if len(boxes) > 0:
            print("✅ Ashoka Pillar detected using YOLO.")
            return "Real,", 90  # Assume 90% confidence from YOLO
        
    except Exception as e:
        print(f"⚠️ YOLO error: {e}")

    print("❌ No Ashoka Pillar matched.")
    return "Fake Ashoka Pillar", 0



#Function RBI_LOGO
def detect_rbi_logo(
    image_path,
    template_paths=[
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RBI_logo.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RBI_logo_1.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RBI_logo_2.png"  # Add second template path
    ],
    threshold=0.55,
    yolo_model_path=r"C:\Users\Apurva Gole\Documents\Data\Data\yolov8n.pt",
    debug=False
):
    import cv2
    import numpy as np
    import os
    from ultralytics import YOLO

    # Load image
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Error loading main image.")
        return ("Fake,", 0, False)

    # Preprocessing function
    def preprocess_for_template_matching(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    img_gray = preprocess_for_template_matching(img)

    # Try matching with each template
    for template_path in template_paths:
        template = cv2.imread(template_path)

        if template is None:
            print(f"❌ Error loading template: {template_path}")
            continue  # Try next template if this one fails

        template_gray = preprocess_for_template_matching(template)

        found = None
        for scale in np.linspace(0.5, 1.5, 10)[::-1]:
            resized_template = cv2.resize(template_gray, (
                int(template_gray.shape[1] * scale),
                int(template_gray.shape[0] * scale)
            ))

            tH, tW = resized_template.shape[:2]

            if img_gray.shape[0] < tH or img_gray.shape[1] < tW:
                continue

            result = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if debug:
                print(f"[{os.path.basename(template_path)}] Scale: {scale:.2f}, Max Match Confidence: {max_val:.3f}")

            if found is None or max_val > found[0]:
                found = (max_val, max_loc, scale, tW, tH)

        if found:
            max_val, max_loc, scale, tW, tH = found
            confidence = min(max_val * 100, 100)
            print(f"🔍 RBI Logo Match Confidence ({os.path.basename(template_path)}): {max_val:.3f}")

            if max_val >= threshold:
                if debug:
                    top_left = max_loc
                    bottom_right = (top_left[0] + tW, top_left[1] + tH)
                    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
                    cv2.imshow(f"Detected RBI Logo: {os.path.basename(template_path)}", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                return ("Real,", confidence, True)

    # 🔁 Fallback: YOLOv8 detection
    print("🔄 Template match failed — trying YOLO RBI logo detection...")

    try:
        yolo_model = YOLO(yolo_model_path)
        results = yolo_model(image_path)
        boxes = results[0].boxes

        if debug:
            print(f"📦 YOLO detected {len(boxes)} objects.")

        best_confidence = 0
        if len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0])  # YOLO box confidence
                if conf > best_confidence:
                    best_confidence = conf

            print(f"✅ RBI Logo detected using YOLO with confidence {best_confidence:.3f}")
            return ("Real,", best_confidence * 100, True)  # Convert to percentage

    except Exception as e:
        print(f"⚠️ YOLO error: {e}")

    print("❌ RBI Logo not detected.")
    return ("Fake,", 0, False)


#Function to check signature
def detect_governor_signature(
    image_path,
    signature_templates=[
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\Signature_1.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\Signature_2.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\Signature_3.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\Signature_4.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\Signature_5.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\Signature_6.png"
    ],
    yolo_model_path=r"C:\Users\Apurva Gole\Documents\Data\Data\yolov8n.pt",
    debug=False
):
    import cv2
    import numpy as np
    import os
    from ultralytics import YOLO

    note_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if note_img is None:
        print("❌ Error loading note image.")
        return "Fake Governor Signature"

    # Enhance image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    note_img = cv2.filter2D(note_img, -1, sharpen_kernel)
    note_img = cv2.equalizeHist(note_img)
    note_img = cv2.fastNlMeansDenoising(note_img, h=10)

    # ORB Matching
    orb = cv2.ORB_create(nfeatures=1000)
    kp_note, des_note = orb.detectAndCompute(note_img, None)

    if des_note is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for template_path in signature_templates:
            if not os.path.exists(template_path):
                print(f"❌ Could not load template: {template_path}")
                continue

            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            kp_temp, des_temp = orb.detectAndCompute(template_img, None)
            if des_temp is None:
                continue

            matches = bf.match(des_temp, des_note)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]
            confidence = min(len(good_matches) * 10, 100)

            if debug:
                print(f"🔍 Matching {template_path} → Good Matches: {len(good_matches)}")
                match_img = cv2.drawMatches(template_img, kp_temp, note_img, kp_note, matches[:20], None, flags=2)
                cv2.imshow(f"Match with {template_path}", match_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if len(good_matches) >= 10:
                print(f"✅ Governor signature detected using {template_path}")
                return "Real,", confidence

    print("🔄 ORB matching failed — trying YOLO signature structure detection...")

    # YOLO fallback
    try:
        yolo_model = YOLO(yolo_model_path)
        results = yolo_model(image_path)
        boxes = results[0].boxes

        if debug:
            print(f"📦 YOLO detected {len(boxes)} objects.")
            #results[0].show()

        if len(boxes) > 0:
            print("✅ Governor signature detected using YOLO.")
            return "Real,", 90  # Assume YOLO detection confidence is 90
        
    except Exception as e:
        print(f"⚠️ YOLO error: {e}")

    print("❌ No Governor signature matched.")
    return "Fake Governor Signature", 0



#Function Mahatma Gandhi 
def detect_gandhi_face(
    image_path,
    gandhi_templates=[
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\mahatma_gandhi_1.jpg",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\mahatma_gandhi_2.jpg.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\mahatma_gandhi_3.jpg.png"  # <- You can add more templates here
    ],
    yolo_model_path=r"C:\Users\Apurva Gole\Documents\Data\Data\yolov8n.pt",
    debug=False
):
    import cv2
    import numpy as np
    import os
    from ultralytics import YOLO

    confidence = 0
    gandhi_detected = False

    # Load note image
    note_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if note_img is None:
        print("❌ Error loading note image.")
        return "Fake,", 0, False

    # Preprocess note image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    note_img = cv2.filter2D(note_img, -1, sharpen_kernel)
    note_img = cv2.equalizeHist(note_img)
    note_img = cv2.fastNlMeansDenoising(note_img, h=10)

    orb = cv2.ORB_create(nfeatures=1000)
    kp_note, des_note = orb.detectAndCompute(note_img, None)

    if des_note is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for template_path in gandhi_templates:
            if not os.path.exists(template_path):
                print(f"❌ Template not found: {template_path}")
                continue

            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            kp_temp, des_temp = orb.detectAndCompute(template_img, None)

            if des_temp is None:
                continue

            matches = bf.match(des_temp, des_note)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]
            confidence = min(len(good_matches) * 10, 100)

            if debug:
                print(f"🔍 Matching {template_path} → Good Matches: {len(good_matches)}")
                if len(kp_temp) > 0 and len(kp_note) > 0:
                    match_img = cv2.drawMatches(template_img, kp_temp, note_img, kp_note, matches[:20], None, flags=2)
                    cv2.imshow(f"Match with {os.path.basename(template_path)}", match_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            if len(good_matches) >= 10:
                print(f"✅ Gandhi face detected using {os.path.basename(template_path)}")
                return "Real,", confidence, True

    # 🔁 YOLO fallback
    print("🔄 ORB matching failed — trying YOLO Gandhi face detection...")

    try:
        yolo_model = YOLO(yolo_model_path)
        results = yolo_model(image_path)
        boxes = results[0].boxes

        if debug:
            print(f"📦 YOLO detected {len(boxes)} objects.")

        best_confidence = 0
        if len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0])  # YOLO box confidence
                if conf > best_confidence:
                    best_confidence = conf

            print(f"✅ Gandhi face detected using YOLO with confidence {best_confidence:.3f}")
            return "Real,", best_confidence * 100, True  # convert to percentage

    except Exception as e:
        print(f"⚠️ YOLO error: {e}")

    print("❌ No Gandhi face matched.")
    return "Fake,", 0, False



#Function for RBI TEXT
import cv2
import numpy as np
import re
import os
from ultralytics import YOLO
from easyocr import Reader
# Function to normalize Hindi text (to remove extra spaces)
def normalize_hindi_text(text):
    return re.sub(r'\s+', '', text)  # Remove any extra spaces

# Function to detect RBI text using OCR
def detect_rbi_text_with_reader(image_path, reader, debug=False):
    confidence = 0  # Initialize confidence
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # OCR process to read text from the image
    result = reader.readtext(gray)
    extracted_text = " ".join([detection[1] for detection in result])

    if debug:
        print("🔍 OCR Detected Text (Raw):")
        for detection in result:
            print(detection)  # Print the detected text with bounding box info.
        print(extracted_text)

    # Clean and normalize the text (remove unwanted characters)
    text_cleaned = re.sub(r'[^\w\s\u0900-\u097F]', '', extracted_text, flags=re.UNICODE).lower()

    # Special case: Handle misrecognized OCR words (e.g., "रि ज़र्व", "रिज़र्व") directly
    text_cleaned = text_cleaned.replace("रि ज़र्व", "रिज़र्व")
    text_cleaned = text_cleaned.replace("रिज़र्व", "रिज़र्व")  # Ensure both variants are fixed
    text_cleaned = text_cleaned.replace("प्र त्याभूत", "प्रत्याभूत")
    text_cleaned = text_cleaned.replace("ग्र त्याभूत", "प्रत्याभूत")
    text_cleaned = text_cleaned.replace("ग्रत्याभूत", "प्रत्याभूत")
    text_cleaned = text_cleaned.replace("प्रत्याभूत", "प्रत्याभूत")  # Ensure both variants are fixed
    text_cleaned = text_cleaned.replace("ooyenvent", "government")
    text_cleaned = text_cleaned.replace("tre", "the")
    text_cleaned = text_cleaned.replace("प्र त्यापर", "प्रत्याभूत")
    text_cleaned = text_cleaned.replace("प्रत्यापर", "प्रत्याभूत")  # Ensure both variants are fixed
    text_cleaned = text_cleaned.replace("दा रार", "द्वारा")
    text_cleaned = text_cleaned.replace("दारार", "द्वारा")  # Ensure both variants are fixed
    text_cleaned = text_cleaned.replace("दा रा", "द्वारा")  # Ensure both variants are fixed
    text_cleaned = text_cleaned.replace("दारा", "द्वारा")  # Ensure both variants are fixed
    text_cleaned = text_cleaned.replace("ररकर", "सरकार") 
    text_cleaned = text_cleaned.replace("र र क र", "सरकार") 
    text_cleaned = text_cleaned.replace("ररकर", "सरकार") 
    text_cleaned = text_cleaned.replace("र र कर", "सरकार")
    text_cleaned = text_cleaned.replace("झ न्द्रीय", "केन्द्रीय") 
    text_cleaned = text_cleaned.replace("झन्द्रीय", "केन्द्रीय") 

    # Separate the Hindi and English text
    words = text_cleaned.split()
    words_hindi = [word for word in words if re.search(r'[\u0900-\u097F]', word)]  # Hindi words
    words_english = [word for word in words if not re.search(r'[\u0900-\u097F]', word)]  # English words

    # Normalize spaces in Hindi words only
    words_hindi = [normalize_hindi_text(word) for word in words_hindi]

    # Reconstruct the cleaned and normalized text
    text_normalized = ' '.join(words_english + words_hindi)

    if debug:
        print(f"Normalized Text: {text_normalized}")

    # Check for the RBI text using prefix/suffix logic
    prefix_1 = prefix_2 = prefix_3 = prefix_4 = prefix_5 = prefix_6 = prefix_7 = prefix_8 = prefix_9 = False
    suffix_1 = suffix_2 = suffix_3 = suffix_4 = suffix_5 = suffix_6 = suffix_7 = False
    match_count = 0  # <==== Important missing initialization

    for word in text_normalized.split():
        print(f"Checking word: {word}")

        if word == "reserve":
            prefix_1 = True
            print("✅ Matched 'RESERVE' as prefix_1")

        elif word == "bank":
            prefix_2 = True
            print("✅ Matched 'BANK' as prefix_2")

        elif word == "of":
            prefix_3 = True
            print("✅ Matched 'OF' as prefix_3")

        elif word == "india":
            prefix_4 = True
            print("✅ Matched 'INDIA' as prefix_4")

        elif word == "guaranteed":
            prefix_5 = True
            print("✅ Matched 'GUARANTEED' as prefix_5")

        elif word == "by":
            prefix_6 = True
            print("✅ Matched 'BY' as prefix_6")

        elif word == "the":
            prefix_7 = True
            print("✅ Matched 'THE' as prefix_7")

        elif word == "central":
            prefix_8 = True
            print("✅ Matched 'CENTRAL' as prefix_8")

        elif word == "government":
            prefix_9 = True
            print("✅ Matched 'GOVERNMENT' as prefix_9")

        elif "भारतीय" in word:
            suffix_1 = True
            print("✅ Matched 'भारतीय' as suffix_1")

        elif "रिज़र्व" in word:
            suffix_2 = True
            print("✅ Matched 'रिज़र्व' as suffix_2")

        elif "बैंक" in word:
            suffix_3 = True
            print("✅ Matched 'बैंक' as suffix_3")

        elif "केन्द्रीय" in word:
            suffix_4 = True
            print("✅ Matched 'केन्द्रीय' as suffix_4")

        elif "सरकार" in word:
            suffix_5 = True
            print("✅ Matched 'सरकार' as suffix_5")

        elif "द्वारा" in word:
            suffix_6 = True
            print("✅ Matched 'द्वारा' as suffix_6")

        elif "प्रत्याभूत" in word:
            suffix_7 = True
            print("✅ Matched 'प्रत्याभूत' as suffix_7")

        if all([prefix_1, prefix_2, prefix_3, prefix_4, prefix_5, prefix_6, prefix_7, prefix_8, prefix_9, suffix_1, suffix_2, suffix_3, suffix_4, suffix_5, suffix_6, suffix_7]):
            confidence = 99.99
            print("✅ RBI Text fully matched using prefix/suffix logic")
            return "Real,", confidence


    # Fallback: search in entire string
    text_lower = text_normalized.lower()
    if not prefix_1 and "reserve" in text_lower:
        prefix_1 = True
        match_count += 1
    if not prefix_2 and "bank" in text_lower:
        prefix_2 = True
        match_count += 1
    if not prefix_3 and "of" in text_lower:
        prefix_3 = True
        match_count += 1
    if not prefix_4 and "india" in text_lower:
        prefix_4 = True
        match_count += 1
    if not prefix_5 and "guaranteed" in text_lower:
        prefix_5 = True
        match_count += 1
    if not prefix_6 and "by" in text_lower:
        prefix_6 = True
        match_count += 1
    if not prefix_7 and "the" in text_lower:
        prefix_7 = True
        match_count += 1
    if not prefix_8 and "central" in text_lower:
        prefix_8 = True
        match_count += 1
    if not prefix_9 and "government" in text_lower:
        prefix_9 = True
        match_count += 1
    if not suffix_1 and "भारतीय" in text_normalized:
        suffix_1 = True
        match_count += 1
    if not suffix_2 and "रिज़र्व" in text_normalized:  # This should now work with 'रिज़र्व'
        suffix_2 = True
        match_count += 1
    if not suffix_3 and "बैंक" in text_normalized:
        suffix_3 = True
        match_count += 1
    if not suffix_4 and "केन्द्रीय" in text_normalized:
        suffix_4 = True
        match_count += 1
    if not suffix_5 and "सरकार" in text_normalized:  # This should now work with 'रिज़र्व'
        suffix_5 = True
        match_count += 1
    if not suffix_6 and "द्वारा" in text_normalized:
        suffix_6 = True
        match_count += 1
    if not suffix_7 and "प्रत्याभूत" in text_normalized:
        suffix_7 = True
        match_count += 1

    if all([prefix_1, prefix_2, prefix_3, prefix_4, prefix_5, prefix_6, prefix_7, prefix_8, prefix_9, suffix_1, suffix_2, suffix_3, suffix_4, suffix_5, suffix_6, suffix_7]):
            confidence = 99.99
            print("✅ RBI Text fully matched using prefix/suffix logic")
            return "Real,", confidence
    
    print("🔄 OCR failed — trying YOLO and ORB fallback...")
    # ==== YOLO fallback ====
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(r"C:\Users\Apurva Gole\Documents\Data\Data\yolov8n.pt")
        results = yolo_model(image_path)  # ✅ Correct

        boxes = results[0].boxes
        if debug:
            print(f"📦 YOLO RBI detections: {len(boxes)}")

        if len(boxes) > 0:
            print("✅ RBI text detected by YOLO.")
            return "Real", 90
    except Exception as e:
        print(f"⚠️ YOLO Error: {e}")

    # ==== ORB Template matching fallback ====
    orb = cv2.ORB_create(nfeatures=1000)
    kp_note, des_note = orb.detectAndCompute(gray, None)
    if des_note is None:
        print("❌ No keypoints found in currency image.")
        return "Fake RBI Text", 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # English templates
    english_templates = [
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_1.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_2.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_3.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_4.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_5.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_6.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_7.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_8.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_9.png"
    ]

    # Hindi templates
    hindi_templates = [
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HINDI.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HIND_1.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HIND_2.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HIND_3.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HIND_4.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HIND_5.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HIND_6.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HIND_6.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HIND_8.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HIND_9.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HIND_10.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\RESERVE_BANK_OF_INDIA_HIND_11.png"
    ]

    def match_template_list(template_list, label="Template"):
        for template_path in template_list:
            if not os.path.exists(template_path):
                print(f"🚫 Missing {label}: {template_path}")
                continue

            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"❌ Could not load {label}: {template_path}")
                continue

            kp_temp, des_temp = orb.detectAndCompute(template, None)
            if des_temp is None:
                continue

            matches = bf.match(des_temp, des_note)
            good = [m for m in matches if m.distance < 50]

            if debug:
                print(f"🧪 Matches with {os.path.basename(template_path)}: {len(good)}")

            if len(good) >= 10:
                print(f"✅ RBI Text detected using {template_path}")
                return "Real", 85  # <- Return label and confidence score directly

        return "Fake RBI Text", 0  # <- Return this if NO matching templates are found


    # Match English templates first
    result_label, result_confidence = match_template_list(english_templates, label="English Template")
    if result_label == "Real":
        return result_label, result_confidence

    # Match Hindi templates if English fails
    result_label, result_confidence = match_template_list(hindi_templates, label="Hindi Template")
    if result_label == "Real":
        return result_label, result_confidence

    # No match found
    confidence = min(match_count * 15, 100)
    print(f"❌ RBI text not detected. Final confidence: {confidence}%")
    return "Fake RBI Text", confidence




# Function to detect denominations written in Hindi/Devanagari or English numbers
def detect_denominaton_hindi(
    image_path,
    denominaton_templates=[
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\10_denomination_1.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\20_denomination_1.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\50_denomination_1.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\100_denomination_1.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\200_denomination_1.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_1.png"
    ],
    yolo_model_path=r"C:\Users\Apurva Gole\Documents\Data\Data\yolov8n.pt",
    debug=False
):
    import cv2
    import numpy as np
    import os
    from ultralytics import YOLO
    confidence = 0  # Initialize confidence
    note_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if note_img is None:
        print("❌ Error loading note image.")
        return "Fake denomination1 code"

    # Enhance image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    note_img = cv2.filter2D(note_img, -1, sharpen_kernel)
    note_img = cv2.equalizeHist(note_img)
    note_img = cv2.fastNlMeansDenoising(note_img, h=10)

    # ORB Matching
    orb = cv2.ORB_create(nfeatures=1000)
    kp_note, des_note = orb.detectAndCompute(note_img, None)

    if des_note is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for template_path in denominaton_templates:
            if not os.path.exists(template_path):
                print(f"❌ Could not load template: {template_path}")
                continue

            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            kp_temp, des_temp = orb.detectAndCompute(template_img, None)
            if des_temp is None:
                continue

            matches = bf.match(des_temp, des_note)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]

            if debug:
                print(f"🔍 Matching {template_path} → Good Matches: {len(good_matches)}")
                match_img = cv2.drawMatches(template_img, kp_temp, note_img, kp_note, matches[:20], None, flags=2)
                cv2.imshow(f"Match with {template_path}", match_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if len(good_matches) >= 10:
                print(f"✅ Denomination1 detected using {template_path}")
                confidence = min(len(good_matches) * 10, 100)
                return "Real,", confidence

    print("🔄 ORB matching failed — trying YOLO denomination structure detection...")

    # YOLO fallback
    try:
        yolo_model = YOLO(yolo_model_path)
        results = yolo_model(image_path)
        boxes = results[0].boxes

        if debug:
            print(f"📦 YOLO detected {len(boxes)} objects.")
            #results[0].show()

        if len(boxes) > 0:
            print("✅ denomination1 detected using YOLO.")
            return "Real,", 90 
    except Exception as e:
        print(f"⚠️ YOLO error: {e}")

    print("❌ No Denomination1 matched.")
    return "Fake Denomination 1", 0

#####################
def detect_denominaton_english(
    image_path,
    denominaton_templates=[
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\10_denomination_2.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\20_denomination_2.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\50_denomination_2.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\100_denomination_2.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\200_denomination_2.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_2.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_23.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_24.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_25.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_26.png"
    ],
    yolo_model_path=r"C:\Users\Apurva Gole\Documents\Data\Data\yolov8n.pt",
    debug=False
):
    import cv2
    import numpy as np
    import os
    from ultralytics import YOLO

    note_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if note_img is None:
        print("❌ Error loading note image.")
        return "Fake denomination2 code"

    # Enhance image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    note_img = cv2.filter2D(note_img, -1, sharpen_kernel)
    note_img = cv2.equalizeHist(note_img)
    note_img = cv2.fastNlMeansDenoising(note_img, h=10)

    # ORB Matching
    orb = cv2.ORB_create(nfeatures=1000)
    kp_note, des_note = orb.detectAndCompute(note_img, None)

    if des_note is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for template_path in denominaton_templates:
            if not os.path.exists(template_path):
                print(f"❌ Could not load template: {template_path}")
                continue

            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            kp_temp, des_temp = orb.detectAndCompute(template_img, None)
            if des_temp is None:
                continue

            matches = bf.match(des_temp, des_note)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]

            if debug:
                print(f"🔍 Matching {template_path} → Good Matches: {len(good_matches)}")
                match_img = cv2.drawMatches(template_img, kp_temp, note_img, kp_note, matches[:20], None, flags=2)
                cv2.imshow(f"Match with {template_path}", match_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if len(good_matches) >= 10:
                print(f"✅ Denomination2 detected using {template_path}")
                confidence = min(len(good_matches) * 10, 100)
                return "Real,", confidence

    print("🔄 ORB matching failed — trying YOLO denomination structure detection...")

    # YOLO fallback
    try:
        yolo_model = YOLO(yolo_model_path)
        results = yolo_model(image_path)
        boxes = results[0].boxes

        if debug:
            print(f"📦 YOLO detected {len(boxes)} objects.")
            #results[0].show()

        if len(boxes) > 0:
            print("✅ denomination2 detected using YOLO.")
            return "Real,", 90
    except Exception as e:
        print(f"⚠️ YOLO error: {e}")

    print("❌ No Denomination2 matched.")
    return "Fake Denomination 2", 0
###############################
def detect_denominaton_english_1(
    image_path,
    denominaton_templates=[
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\10_denomination_3.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\20_denomination_3.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\50_denomination_3.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\100_denomination_3.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\200_denomination_3.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_3.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_31.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_32.png"
    ],
    yolo_model_path=r"C:\Users\Apurva Gole\Documents\Data\Data\yolov8n.pt",
    debug=False
):
    import cv2
    import numpy as np
    import os
    from ultralytics import YOLO

    note_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if note_img is None:
        print("❌ Error loading note image.")
        return "Fake denomination3 code"

    # Enhance image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    note_img = cv2.filter2D(note_img, -1, sharpen_kernel)
    note_img = cv2.equalizeHist(note_img)
    note_img = cv2.fastNlMeansDenoising(note_img, h=10)

    # ORB Matching
    orb = cv2.ORB_create(nfeatures=1000)
    kp_note, des_note = orb.detectAndCompute(note_img, None)

    if des_note is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for template_path in denominaton_templates:
            if not os.path.exists(template_path):
                print(f"❌ Could not load template: {template_path}")
                continue

            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            kp_temp, des_temp = orb.detectAndCompute(template_img, None)
            if des_temp is None:
                continue

            matches = bf.match(des_temp, des_note)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]

            if debug:
                print(f"🔍 Matching {template_path} → Good Matches: {len(good_matches)}")
                match_img = cv2.drawMatches(template_img, kp_temp, note_img, kp_note, matches[:20], None, flags=2)
                cv2.imshow(f"Match with {template_path}", match_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if len(good_matches) >= 10:
                print(f"✅ Denomination3 detected using {template_path}")
                confidence = min(len(good_matches) * 10, 100)
                return "Real,", confidence

    print("🔄 ORB matching failed — trying YOLO denomination structure detection...")

    # YOLO fallback
    try:
        yolo_model = YOLO(yolo_model_path)
        results = yolo_model(image_path)
        boxes = results[0].boxes

        if debug:
            print(f"📦 YOLO detected {len(boxes)} objects.")
            #results[0].show()

        if len(boxes) > 0:
            print("✅ denomination3 detected using YOLO.")
            return "Real,", 90
    except Exception as e:
        print(f"⚠️ YOLO error: {e}")

    print("❌ No Denomination3 matched.")
    return "Fake Denomination 3", 0
################################
def detect_denominaton_english_2(
    image_path,
    denominaton_templates=[
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\10_denomination_4.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\20_denomination_4.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\50_denomination_4.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\100_denomination_4.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\200_denomination_4.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_4.png"
    ],
    yolo_model_path=r"C:\Users\Apurva Gole\Documents\Data\Data\yolov8n.pt",
    debug=False
):
    import cv2
    import numpy as np
    import os
    from ultralytics import YOLO

    note_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if note_img is None:
        print("❌ Error loading note image.")
        return "Fake denomination4 code"

    # Enhance image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    note_img = cv2.filter2D(note_img, -1, sharpen_kernel)
    note_img = cv2.equalizeHist(note_img)
    note_img = cv2.fastNlMeansDenoising(note_img, h=10)

    # ORB Matching
    orb = cv2.ORB_create(nfeatures=1000)
    kp_note, des_note = orb.detectAndCompute(note_img, None)

    if des_note is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for template_path in denominaton_templates:
            if not os.path.exists(template_path):
                print(f"❌ Could not load template: {template_path}")
                continue

            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            kp_temp, des_temp = orb.detectAndCompute(template_img, None)
            if des_temp is None:
                continue

            matches = bf.match(des_temp, des_note)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]

            if debug:
                print(f"🔍 Matching {template_path} → Good Matches: {len(good_matches)}")
                match_img = cv2.drawMatches(template_img, kp_temp, note_img, kp_note, matches[:20], None, flags=2)
                cv2.imshow(f"Match with {template_path}", match_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if len(good_matches) >= 10:
                print(f"✅ Denomination4 detected using {template_path}")
                confidence = min(len(good_matches) * 10, 100)
                return "Real,", confidence

    print("🔄 ORB matching failed — trying YOLO denomination structure detection...")

    # YOLO fallback
    try:
        yolo_model = YOLO(yolo_model_path)
        results = yolo_model(image_path)
        boxes = results[0].boxes

        if debug:
            print(f"📦 YOLO detected {len(boxes)} objects.")
            #results[0].show()

        if len(boxes) > 0:
            print("✅ denomination4 detected using YOLO.")
            return "Real,", 90
    except Exception as e:
        print(f"⚠️ YOLO error: {e}")

    print("❌ No Denomination4 matched.")
    return "Fake Denomination 4", 0
#################################
def detect_denominaton_english_3(
    image_path,
    denominaton_templates=[
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\10_denomination_5.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\20_denomination_5.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\50_denomination_5.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\100_denomination_5.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\200_denomination_5.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_5.png"
    ],
    yolo_model_path=r"C:\Users\Apurva Gole\Documents\Data\Data\yolov8n.pt",
    debug=False
):
    import cv2
    import numpy as np
    import os
    from ultralytics import YOLO

    note_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if note_img is None:
        print("❌ Error loading note image.")
        return "Fake denomination5 code"

    # Enhance image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    note_img = cv2.filter2D(note_img, -1, sharpen_kernel)
    note_img = cv2.equalizeHist(note_img)
    note_img = cv2.fastNlMeansDenoising(note_img, h=10)

    # ORB Matching
    orb = cv2.ORB_create(nfeatures=1000)
    kp_note, des_note = orb.detectAndCompute(note_img, None)

    if des_note is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for template_path in denominaton_templates:
            if not os.path.exists(template_path):
                print(f"❌ Could not load template: {template_path}")
                continue

            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            kp_temp, des_temp = orb.detectAndCompute(template_img, None)
            if des_temp is None:
                continue

            matches = bf.match(des_temp, des_note)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]

            if debug:
                print(f"🔍 Matching {template_path} → Good Matches: {len(good_matches)}")
                match_img = cv2.drawMatches(template_img, kp_temp, note_img, kp_note, matches[:20], None, flags=2)
                cv2.imshow(f"Match with {template_path}", match_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if len(good_matches) >= 10:
                print(f"✅ Denomination5 detected using {template_path}")
                confidence = min(len(good_matches) * 10, 100)
                return "Real,", confidence

    print("🔄 ORB matching failed — trying YOLO denomination structure detection...")

    # YOLO fallback
    try:
        yolo_model = YOLO(yolo_model_path)
        results = yolo_model(image_path)
        boxes = results[0].boxes

        if debug:
            print(f"📦 YOLO detected {len(boxes)} objects.")
            #results[0].show()

        if len(boxes) > 0:
            print("✅ denomination5 detected using YOLO.")
            return "Real,", 90
    except Exception as e:
        print(f"⚠️ YOLO error: {e}")

    print("❌ No Denomination5 matched.")
    return "Fake Denomination 5", 0
###########################
def detect_denominaton_english_4(
    image_path,
    denominaton_templates=[
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\100_denomination_6.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\200_denomination_6.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\200_denomination_7.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\200_denomination_8.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_6.png",
        r"C:\Users\Apurva Gole\Documents\Data\Data\DATASET4\500_denomination_61.png"
    ],
    yolo_model_path=r"C:\Users\Apurva Gole\Documents\Data\Data\yolov8n.pt",
    debug=False
):
    import cv2
    import numpy as np
    import os
    from ultralytics import YOLO

    note_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if note_img is None:
        print("❌ Error loading note image.")
        return "Fake denomination5 code"

    # Enhance image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    note_img = cv2.filter2D(note_img, -1, sharpen_kernel)
    note_img = cv2.equalizeHist(note_img)
    note_img = cv2.fastNlMeansDenoising(note_img, h=10)

    # ORB Matching
    orb = cv2.ORB_create(nfeatures=1000)
    kp_note, des_note = orb.detectAndCompute(note_img, None)

    if des_note is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for template_path in denominaton_templates:
            if not os.path.exists(template_path):
                print(f"❌ Could not load template: {template_path}")
                continue

            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            kp_temp, des_temp = orb.detectAndCompute(template_img, None)
            if des_temp is None:
                continue

            matches = bf.match(des_temp, des_note)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]

            if debug:
                print(f"🔍 Matching {template_path} → Good Matches: {len(good_matches)}")
                match_img = cv2.drawMatches(template_img, kp_temp, note_img, kp_note, matches[:20], None, flags=2)
                cv2.imshow(f"Match with {template_path}", match_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if len(good_matches) >= 10:
                print(f"✅ Denomination6 detected using {template_path}")
                confidence = min(len(good_matches) * 10, 100)
                return "Real,", confidence

    print("🔄 ORB matching failed — trying YOLO denomination structure detection...")

    # YOLO fallback
    try:
        yolo_model = YOLO(yolo_model_path)
        results = yolo_model(image_path)
        boxes = results[0].boxes

        if debug:
            print(f"📦 YOLO detected {len(boxes)} objects.")
            #results[0].show()

        if len(boxes) > 0:
            print("✅ denomination6 detected using YOLO.")
            return "Real,", 90
    except Exception as e:
        print(f"⚠️ YOLO error: {e}")

    print("❌ No Denomination6 matched.")
    return "Fake Denomination 6", 0

def detect_denomination_code(image_path, reader=None, debug=False):
    hindi_result, hindi_conf = detect_denominaton_hindi(image_path, debug=debug)
    english_result, english_conf = detect_denominaton_english(image_path, debug=debug)
    english1_result, english1_conf = detect_denominaton_english_1(image_path, debug=debug)
    english2_result, english2_conf = detect_denominaton_english_2(image_path, debug=debug)
    english3_result, english3_conf = detect_denominaton_english_3(image_path, debug=debug)
    english4_result, english4_conf = detect_denominaton_english_4(image_path, debug=debug)

    results = {
        "Hindi": (hindi_result.strip(), hindi_conf),
        "English": (english_result.strip(), english_conf),
        "English1": (english1_result.strip(), english1_conf),
        "English2": (english2_result.strip(), english2_conf),
        "English3": (english3_result.strip(), english3_conf),
        "English4": (english4_result.strip(), english4_conf)
    }

    real_matches = [key for key, (val, _) in results.items() if val.startswith("Real")]

    print("\n📊 Denomination Detection Summary:")
    for key, (val, conf) in results.items():
        print(f"  {key}: {val} (Confidence: {conf}%)")

    # Calculate overall confidence
    if len(real_matches) >= 4:
        avg_confidence = int(np.mean([conf for key, (val, conf) in results.items() if val.startswith("Real")]))
        print(f"✅ Detected denomination from {len(real_matches)} features with Avg Confidence: {avg_confidence}%")
        return "Real", avg_confidence
    else:
        avg_confidence = int(np.mean([conf for key, (val, conf) in results.items()]))
        print(f"❌ Denomination not detected confidently (less than 4 features matched). Avg Confidence: {avg_confidence}%")
        return "Fake Denomination", avg_confidence

#Function for All
def check_important_features(image_path, reader, debug=True):
    """
    Integrates all feature checks: Ashoka Pillar, RBI logo, RBI text, Gandhi image,
    Governor signature, denomination code (Rxx), and uses them alongside ML/CNN.

    Returns:
        bool: True if all important features are detected, else False.
    """
    print("\n🧪 Checking important currency features...")

    # Perform individual feature checks
    # Ashoka Pillar
    ashoka_result = detect_ashoka_pillar(image_path)
    ashoka_ok = ashoka_result[0] == "Real,"

    # RBI Logo
    rbi_logo_result = detect_rbi_logo(image_path)
    rbi_logo_ok = rbi_logo_result[0] == "Real," if isinstance(rbi_logo_result, tuple) else rbi_logo_result

    # RBI Text
    rbi_text_result, _ = detect_rbi_text_with_reader(image_path, reader, debug=True)
    rbi_text_ok = rbi_text_result == "Real,"

    # Gandhi face
    gandhi_result, _, _ = detect_gandhi_face(image_path)
    gandhi_ok = gandhi_result == "Real,"

    # Governor signature
    signature_result, _ = detect_governor_signature(image_path)
    signature_ok = signature_result == "Real,"

    # Denomination 
    denomination_result = detect_denomination_code(image_path, reader, False)
    denomination_ok = denomination_result != "Denomination Not Found"

    #serial number
    serial_number = extract_serial_number(image_path, debug=True)
    if serial_number != "Fake Serial Number":
        serial_ok = "Real"  # ✅ Assign properly
    else:
        serial_ok = "Fake"
    # Master Feature Check 🔥
    all_ok = all([
        ashoka_ok,
        rbi_logo_ok,
        rbi_text_ok,
        gandhi_ok,
        signature_ok,
        denomination_ok,
        serial_ok  # ✅ Include serial number validation also
    ])

    if all_ok:
        final_status = "Real"
        print("✅ All important features detected! Status: Real")
    else:
        final_status = "Fake"
        print("❌ One or more important features missing. Status: Fake")
        # List missing features
        missing_features = []
        if not ashoka_ok:
            missing_features.append("Ashoka Pillar")
        if not rbi_logo_ok:
            missing_features.append("RBI Logo")
        if not rbi_text_ok:
            missing_features.append("RBI Text")
        if not gandhi_ok:
            missing_features.append("Gandhi Image")
        if not signature_ok:
            missing_features.append("Governor Signature")
        if not denomination_ok:
            missing_features.append("Denomination Code")
        if serial_ok != "Real":
            missing_features.append("Serial Number")

        print(f"🚨 Missing or invalid features: {', '.join(missing_features)}")
        
    # Return results for further analysis or debugging
    feature_results = {
        "ashoka_pillar": ashoka_ok,
        "rbi_logo": rbi_logo_ok,
        "rbi_text": rbi_text_ok,
        "gandhi_image": gandhi_ok,
        "governor_signature": signature_ok,
        "denomination_code": denomination_ok,
        "serial_valid": serial_ok,
        "all_features_ok": all_ok,
        "final_status": final_status
    }

    return feature_results


# FUNCTION TO TRAIN MODEL
def train_models():
    X, y = [], []

    REAL_BASE = os.path.join(DATASET_PATH, "Real Notes")
    FAKE_BASE = os.path.join(DATASET_PATH, "Fake Notes")

    # ✅ Collect Real Images
    for root, dirs, files in os.walk(REAL_BASE):
        if "dataset" in root.lower():
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)

                    # Step 1: Validate with important features and serial
                    serial_number = extract_serial_number(image_path, debug=False)
                    features_ok = check_important_features(image_path, reader, debug=False)
                    

                    if features_ok["all_features_ok"] and serial_number != "Fake Serial Number":
                        features = extract_features(image_path)
                        X.append(features)
                        y.append("real")
                    else:
                        features = extract_features(image_path)
                        X.append(features)
                        y.append("real")

    # ✅ Collect Fake Images
    for root, dirs, files in os.walk(FAKE_BASE):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)

                serial_number = extract_serial_number(image_path, debug=False)
                features_ok = check_important_features(image_path, reader, debug=False)

                if features_ok["all_features_ok"] and serial_number != "Fake Serial Number":
                    features = extract_features(image_path)
                    X.append(features)
                    y.append("real")  # ✅ Considered real due to strong feature + serial match
                else:
                    features = extract_features(image_path)
                    X.append(features)
                    y.append("fake")

    if len(X) == 0:
        print("❌ No images with valid features found for training.")
        return

    X = np.array(X)
    y_encoded = LabelEncoder().fit_transform(y)

    # Save real and fake feature sets separately
    with open("real_features.pkl", 'wb') as f:
        pickle.dump(X[y_encoded == 1], f)
    with open("fake_features.pkl", 'wb') as f:
        pickle.dump(X[y_encoded == 0], f)

    # Split for training/testing
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train models
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    # Save models
    pickle.dump(knn, open("knn_model.pkl", "wb"))
    pickle.dump(rf, open("rf_model.pkl", "wb"))
    
    print("✅ ML Models trained and saved with feature-aware labeling.")

#function of cnn model
def train_cnn_model():
    image_data = []
    labels = []

    real_base = os.path.join(DATASET_PATH, "Real Notes")
    fake_base = os.path.join(DATASET_PATH, "Fake Notes")

    # ✅ Load and filter Real Notes
    for root, dirs, files in os.walk(real_base):
        if "dataset" in root.lower():
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(root, file)
                    serial = extract_serial_number(img_path, debug=False)
                    features_ok = check_important_features(img_path, reader, debug=False)

                    label = "real" if features_ok["all_features_ok"] and serial != "Fake Serial Number" else "real"

                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (224, 224))
                    img = img_to_array(img)
                    img = preprocess_input(img)

                    image_data.append(img)
                    labels.append(label)

    # ✅ Load and filter Fake Notes
    for root, dirs, files in os.walk(fake_base):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, file)
                serial = extract_serial_number(img_path, debug=False)
                features_ok = check_important_features(img_path, reader, debug=False)
                

                label = "real" if features_ok["all_features_ok"] and serial != "Fake Serial Number" else "fake"

                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                img = img_to_array(img)
                img = preprocess_input(img)

                image_data.append(img)
                labels.append(label)

    if len(image_data) == 0:
        print("❌ No valid images found for CNN training.")
        return

    # Prepare data
    X = np.array(image_data, dtype="float32")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    y = to_categorical(y, num_classes=2)

    # Split dataset
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build MobileNetV2 Model
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(2, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)

    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32,
              callbacks=[EarlyStopping(patience=5)])

    model.save("currency_classifier_model.keras")
    print("✅ CNN Model trained and saved with feature-aware logic.")

    
from PIL import Image

# ✅ Save Grayscale image
def save_grayscale_image(image_path, save_path="processed_gray.jpg"):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_path, gray)

# ✅ Save Edge Detection image
def save_edge_image(image_path, save_path="processed_edges.jpg"):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(save_path, edges)

# ✅ Save CNN Preprocessed image
def save_cnn_preprocessed_image(image_path, save_path="processed_cnn.jpg"):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    normalized = image.astype(np.float32) / 255.0
    cv2.imwrite(save_path, (normalized * 255).astype(np.uint8))

# ✅ Combo function for GUI to call all
def extract_and_save_feature_images(image_path):
    save_grayscale_image(image_path)
    save_edge_image(image_path)
    save_cnn_preprocessed_image(image_path)
    print("✅ Processed images saved.")

def predict_currency(image_path):
    # Load models
    cnn_model = load_model("currency_classifier_model.keras")
    knn = pickle.load(open("knn_model.pkl", "rb"))
    rf = pickle.load(open("rf_model.pkl", "rb"))

    # OCR Reader
    reader = easyocr.Reader(['en', 'hi'])

    # Extract serial number
    serial = extract_serial_number(image_path, debug=True)
    if serial == "Fake Serial Number":
        return "Fake - Invalid Serial"

    # Run feature check
    feature_results = check_important_features(image_path, reader, debug=True)

    # Bleed line detection
    bleed_lines = detect_bleed_lines(image_path)
    note_type = get_note_type_by_bleed_lines(bleed_lines)
    print(f"Note Type: {note_type} based on {bleed_lines} bleed lines")

    # ✅ If all key features are present and serial is valid → mark as Real
    if serial != "Fake Serial Number" and feature_results["all_features_ok"]:
        return f"Real - Serial: {serial} - Note Type: {note_type}"

    # Extract features and get ML predictions
    features = extract_features(image_path).reshape(1, -1)
    knn_pred = knn.predict(features)[0]
    rf_pred = rf.predict(features)[0]
    ml_preds = [knn_pred, rf_pred]
    ml_final = max(set(ml_preds), key=ml_preds.count)

    # CNN prediction
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = preprocess_image(image.astype(np.float32))
    cnn_input = np.expand_dims(image, axis=-1)
    cnn_input = np.repeat(cnn_input, 3, axis=-1)
    cnn_input = np.expand_dims(cnn_input, axis=0)
    cnn_pred = cnn_model.predict(cnn_input)[0]
    cnn_label = np.argmax(cnn_pred)

    # Final fallback decision
    if ml_final == 1 and cnn_label == 1:
        return f"Real - Serial: {serial} - Note Type: {note_type}"
    else:
        return f"Fake - Serial: {serial} - Mismatch Detected"
    

# ✅ Check if models exist, else train
if not os.path.exists("knn_model.pkl") or not os.path.exists("rf_model.pkl"):
    print("No trained ML models found. Training now...")
    train_models()
else:
    print("Trained ML models found!")

if not os.path.exists("currency_classifier_model.keras"):
    print("No trained CNN model found. Training now...")
    train_cnn_model()
else:
    print("Trained CNN model found!")

#function for confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import numpy as np
import easyocr

def generate_ml_confusion_matrix():
    matrix_path = "ml_confusion_matrix.png"

    if os.path.exists(matrix_path):
        # 📸 Confusion matrix image already exists, display it
        img = plt.imread(matrix_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title("ML Model (KNN + RF) Confusion Matrix (Saved)")
        plt.show()
    else:
        # 🛠 Confusion matrix not found, generate it
        knn_model = pickle.load(open("knn_model.pkl", "rb"))
        rf_model = pickle.load(open("rf_model.pkl", "rb"))
        y_true_ml, y_pred_ml = [], []

        for label_type, label_value in [("Real Notes", 1), ("Fake Notes", 0)]:
            note_path = os.path.join(DATASET_PATH, label_type)
            for root, dirs, files in os.walk(note_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(root, file)

                        features = extract_features(image_path)
                        knn_pred = knn_model.predict([features])[0]
                        rf_pred = rf_model.predict([features])[0]
                        final_pred = max(set([knn_pred, rf_pred]), key=[knn_pred, rf_pred].count)

                        y_true_ml.append(label_value)
                        y_pred_ml.append(final_pred)

        cm = confusion_matrix(y_true_ml, y_pred_ml, labels=[1, 0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
        disp.plot(cmap='Reds')
        plt.title("ML Model (KNN + RF) Confusion Matrix (Real/Fake Order)")
        plt.grid(False)
        for text in disp.figure_.axes[0].texts:
            text.set_color('black')

        plt.savefig(matrix_path)  # 📸 Save confusion matrix
        plt.show()


def generate_cnn_confusion_matrix():
    matrix_path = "cnn_confusion_matrix.png"

    if os.path.exists(matrix_path):
        # 📸 Confusion matrix image already exists, display it
        img = plt.imread(matrix_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title("CNN Model Confusion Matrix (Saved)")
        plt.show()
    else:
        # 🛠 Confusion matrix not found, generate it
        cnn_model = load_model("currency_classifier_model.keras")
        y_true_cnn, y_pred_cnn = [], []

        for label_type, label_value in [("Real Notes", 1), ("Fake Notes", 0)]:
            note_path = os.path.join(DATASET_PATH, label_type)
            for root, dirs, files in os.walk(note_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(root, file)

                        img = preprocess_image(image_path)
                        cnn_input = np.expand_dims(img, axis=-1)
                        cnn_input = np.repeat(cnn_input, 3, axis=-1)
                        cnn_input = np.expand_dims(cnn_input, axis=0)
                        
                        pred = cnn_model.predict(cnn_input)[0]
                        pred_label = np.argmax(pred)

                        y_true_cnn.append(label_value)
                        y_pred_cnn.append(pred_label)

        cm = confusion_matrix(y_true_cnn, y_pred_cnn, labels=[1, 0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
        disp.plot(cmap='Greens')
        plt.title("CNN Model Confusion Matrix (Real/Fake Order)")
        plt.grid(False)
        for text in disp.figure_.axes[0].texts:
            text.set_color('black')

        plt.savefig(matrix_path)  # 📸 Save confusion matrix
        plt.show()

# function for confidence graph 
import matplotlib.pyplot as plt

def analyze_and_plot_feature_confidences(image_path):
    # Initialize reader for OCR
    reader = easyocr.Reader(['en'])

    # Collect confidences
    feature_confidences = {}

    # Detect features and store confidence
    _, ashoka_conf = detect_ashoka_pillar(image_path)
    feature_confidences['Ashoka Pillar'] = ashoka_conf

    _, rbi_logo_conf, _ = detect_rbi_logo(image_path)
    feature_confidences['RBI Logo'] = rbi_logo_conf

    _, rbi_text_conf = detect_rbi_text_with_reader(image_path, reader, debug=False)
    feature_confidences['RBI Text'] = rbi_text_conf

    _, gandhi_conf, _ = detect_gandhi_face(image_path)
    feature_confidences['Gandhi Face'] = gandhi_conf

    _, signature_conf = detect_governor_signature(image_path)
    feature_confidences['Governor Signature'] = signature_conf

    _, denomination_conf = detect_denomination_code(image_path, reader, debug=False)
    feature_confidences['Denomination'] = denomination_conf

    # (Serial number confidence isn't numerical, so we treat it binary)
    serial_number = extract_serial_number(image_path, debug=False)
    if serial_number != "Fake Serial Number":
        feature_confidences['Serial Number'] = 90
    else:
        feature_confidences['Serial Number'] = 0

    # Plotting
    features = list(feature_confidences.keys())
    confidences = list(feature_confidences.values())

    plt.figure(figsize=(12, 6))
    plt.plot(features, confidences, marker='o', linestyle='-', color='b')
    plt.title('Feature Confidences for Currency Note')
    plt.xlabel('Feature')
    plt.ylabel('Confidence (%)')
    plt.ylim(0, 110)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Also return the dictionary
    return feature_confidences

# ✅ Predict with Models
def predict_with_models(image_path):
    try:
        cnn_model = load_model("currency_classifier_model.keras")
        knn_model = pickle.load(open("knn_model.pkl", "rb"))
        rf_model = pickle.load(open("rf_model.pkl", "rb"))
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return "❌ Model loading failed"

    # ✅ Step 1: Run important feature check and serial number extraction
    reader = easyocr.Reader(['en', 'hi'])
    feature_results = check_important_features(image_path, reader, debug=True)
    serial_number = extract_serial_number(image_path, debug=True)

    # ✅ Step 2: Extract image features for ML
    features = extract_features(image_path).reshape(1, -1)
    knn_pred = knn_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    ml_preds = [knn_pred, rf_pred]
    ml_final = max(set(ml_preds), key=ml_preds.count)

    knn_conf = knn_model.predict_proba(features).max() * 100
    rf_conf = rf_model.predict_proba(features).max() * 100
    ml_conf = round((knn_conf + rf_conf) / 2, 2)

    # ✅ Step 3: CNN Prediction
    cnn_input = preprocess_image(image_path)
    cnn_input = np.expand_dims(cnn_input, axis=-1)
    cnn_input = np.repeat(cnn_input, 3, axis=-1)
    cnn_input = np.expand_dims(cnn_input, axis=0)

    cnn_pred = cnn_model.predict(cnn_input)[0]
    cnn_label = np.argmax(cnn_pred)
    cnn_conf = cnn_pred.max() * 100

    # ✅ Step 4: Bleed line & denomination detection
    is_real, bleed_count = detect_bleed_lines(image_path)
    note_type = get_note_type_by_bleed_lines(bleed_count)

    # ✅ Step 5: Decision Logic
    if serial_number == "Fake Serial Number":
        status = "Fake"
        ml_label = "Fake"
        cnn_label_text = "Fake"
    elif serial_number != "Fake Serial Number" and feature_results["all_features_ok"]:
        status = "Real"
        ml_label = "Real"
        cnn_label_text = "Real"
    else:
        ml_label = "Real" if ml_final == 1 else "Fake"
        cnn_label_text = "Real" if cnn_label == 1 else "Fake"
        status = "Real" if ml_label == "Real" and cnn_label_text == "Real" else "Fake"

    # ✅ Feature Summary Text
    feature_summary = (
        f"Ashoka Pillar: {'✅' if feature_results['ashoka_pillar'] else '❌'}\n"
        f"RBI Logo: {'✅' if feature_results['rbi_logo'] else '❌'}\n"
        f"RBI Text (Hindi/English): {'✅' if feature_results['rbi_text'] else '❌'}\n"
        f"Gandhi Image: {'✅' if feature_results['gandhi_image'] else '❌'}\n"
        f"Governor Signature: {'✅' if feature_results['governor_signature'] else '❌'}\n"
        f"Denomination Code: {'✅' if feature_results['denomination_code'] else '❌'}"
    )

    return {
        "status": status,
        "serial": serial_number,
        "ml_prediction": ml_label,
        "cnn_prediction": cnn_label_text,
        "ml_confidence": ml_conf,
        "cnn_confidence": cnn_conf,
        "bleed_lines": bleed_count,
        "note_type": note_type,
        "feature_checks": feature_results,
        "feature_summary": feature_summary
    }
#History display 
def log_scan_history_text(image_path, result_dict, history_file="scan_history.txt"):
    """
    Logs the image prediction in a table-style plain text file (no CSV),
    with Denomination Type instead of Timestamp.
    """
    denomination = f"₹{result_dict.get('note_type', 'Unknown')}"  # Like ₹500, ₹100 etc.

    # Format one line
    line = "{:<8} {:<20} {:<6} {:<15} {:<10} {:<8} {:<10} {:<8} {:<5}".format(
        denomination,
        os.path.basename(image_path),
        result_dict.get("status", ""),
        result_dict.get("serial", ""),
        result_dict.get("ml_prediction", ""),
        f"{result_dict.get('ml_confidence', 0):.2f}",
        result_dict.get("cnn_prediction", ""),
        f"{result_dict.get('cnn_confidence', 0):.2f}",
        result_dict.get("bleed_lines", "")
    )

    # Create header if file does not exist
    if not os.path.exists(history_file):
        header = "{:<8} {:<20} {:<6} {:<15} {:<10} {:<8} {:<10} {:<8} {:<5}".format(
            "Denom.", "Image Name", "Status", "Serial Number",
            "ML Pred", "ML Conf", "CNN Pred", "CNN Conf", "Lines"
        )
        with open(history_file, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write("=" * len(header) + "\n")

    # Append the result
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    print("📝 Logged to scan_history.txt (with denomination)")

def load_ml_models():
    try:
        knn = pickle.load(open("knn_model.pkl", "rb"))
        rf = pickle.load(open("rf_model.pkl", "rb"))
        print("✅ ML models loaded.")
        return knn, rf
    except Exception as e:
        print(f"❌ Failed to load ML models: {e}")
        return None

def load_cnn_model():
    try:
        cnn_model = load_model("currency_classifier_model.keras")
        print("✅ CNN model loaded successfully.")
        return cnn_model
    except Exception as e:
        print(f"❌ Failed to load CNN model: {e}")
        return None

# ✅ Check if ML and CNN models exist, else train them
if not os.path.exists("knn_model.pkl") or not os.path.exists("rf_model.pkl"):
    print("❌ ML models not found. Training now...")
    train_models()
else:
    print("✅ ML models found!")

if not os.path.exists("currency_classifier_model.keras"):
    print("❌ CNN model not found. Training now...")
    train_cnn_model()
else:
    print("✅ CNN model found!")

# ✅ Function to extract total features from uploaded image
def extract_total_features(image_path):
    features = extract_features(image_path)
    feature_count = len(features)
    print(f"✅ Total Features Extracted from Uploaded Image: {feature_count}")
    return feature_count

def retrain_model_from_image(image_path):
    print(f"📥 Retraining with new image: {image_path}")

    # === 🔁 Load or initialize retrain count ===
    count_file = "retrain_count.txt"
    if os.path.exists(count_file):
        with open(count_file, "r") as f:
            retrain_count = int(f.read().strip()) + 1
    else:
        retrain_count = 1

    # Save updated count
    with open(count_file, "w") as f:
        f.write(str(retrain_count))

    print(f"🔢 Retraining Iteration: {retrain_count}")

    # === 📦 Load existing data ===
    if os.path.exists("real_features.pkl"):
        with open("real_features.pkl", 'rb') as f:
            real_data = list(pickle.load(f))
    else:
        real_data = []

    if os.path.exists("fake_features.pkl"):
        with open("fake_features.pkl", 'rb') as f:
            fake_data = list(pickle.load(f))
    else:
        fake_data = []

    # === 🧠 Determine label for new image ===
    features_ok = check_important_features(image_path, reader, debug=True)
    serial = extract_serial_number(image_path, debug=True)
    features = extract_features(image_path)

    if features_ok["all_features_ok"] and serial != "Fake Serial Number":
        label = "real"
        real_data.append(features)
    else:
        label = "fake"
        fake_data.append(features)

    print(f"🧠 Label determined for retraining: {label.upper()}")

    # === 🧩 Combine and encode ===
    X = np.array(real_data + fake_data)
    y = np.array(["real"] * len(real_data) + ["fake"] * len(fake_data))
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save updated features
    with open("real_features.pkl", 'wb') as f:
        pickle.dump(np.array(real_data), f)
    with open("fake_features.pkl", 'wb') as f:
        pickle.dump(np.array(fake_data), f)

    # === 🔄 Retrain ML models ===
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pickle.dump(knn, open("knn_model.pkl", "wb"))

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    pickle.dump(rf, open("rf_model.pkl", "wb"))

    print("✅ ML models retrained and saved.")

    # === 🧠 Retrain CNN ===
    train_cnn_model()
    print("✅ CNN model retrained and saved.")

    
