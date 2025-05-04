import cv2
import pytesseract
import re

# Set up Tesseract OCR path (Update this for your OS)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update if needed

# List of some known valid RBI prefixes (extend as needed)
VALID_PREFIXES = {"OWC", "XYZ", "ABC", "PQR", "DEF"}  # Add more as needed

def validate_serial_number(image_path):
    """
    Extracts and validates the serial number from an Indian currency note.
    Checks:
      ✅ If the top-left and bottom-right serial numbers match.
      ✅ If the serial format is valid (AAA 123456 or 1AA 123456).
      ✅ If the prefix is a known RBI-issued prefix.
      ✅ If the note is a valid "Star (*) note".
    
    Args:
        image_path (str): Path to the currency note image.

    Returns:
        str: Validation result message.
    """
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        return "❌ Error: Image not found!"

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding for better OCR detection
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Extract text using OCR
    extracted_text = pytesseract.image_to_string(processed, config='--psm 6')

    # Define valid serial number format (Three letters + six digits, 1 digit + two letters + six digits, or Star Note)
    serial_pattern = r'([A-Z]{3} ?\d{6}|\*[A-Z]{3} ?\d{6}|[0-9]{1}[A-Z]{2} ?\d{6})'  

    # Find serial numbers in the extracted text
    serial_numbers = re.findall(serial_pattern, extracted_text)

    # Validate extracted serial numbers
    if len(serial_numbers) >= 2:  # At least two serial numbers should be detected
        top_left_serial = serial_numbers[0]
        bottom_right_serial = serial_numbers[1]

        print(f"🔹 Top Left Serial: {top_left_serial}")
        print(f"🔹 Bottom Right Serial: {bottom_right_serial}")

        # Check if both serial numbers match
        if top_left_serial != bottom_right_serial:
            return "⚠️ Mismatch in serial numbers! This note **could be fake**."

        # Validate Serial Number Format
        if not re.match(r'^[A-Z]{3} \d{6}$', top_left_serial) and not re.match(r'^\*[A-Z]{3} \d{6}$', top_left_serial) and not re.match(r'^[0-9]{1}[A-Z]{2} \d{6}$', top_left_serial):
            return "⚠️ Invalid serial number format! This note may be **fake**."

        # Extract Prefix (3-letter or 1-digit + 2-letter)
        prefix = top_left_serial[:3] if top_left_serial[0].isalpha() else top_left_serial[:2]

        # Validate Prefix
        if "*" not in prefix and prefix not in VALID_PREFIXES:
            return f"⚠️ Suspicious Prefix `{prefix}`! This may be a **fake note**."

        # Detect Star (*) Notes Misuse
        if "*" in top_left_serial:
            if prefix.replace("*", "") not in VALID_PREFIXES:
                return f"⚠️ Incorrect Star Note `{top_left_serial}`! Possible **fake note misuse**."

        return "✅ The serial numbers match and appear **valid**."

    return "❌ Unable to detect valid serial numbers. The note may be **fake** or OCR failed."

# Example usage
if __name__ == "__main__":
    image_path = "your_currency_note_image.jpg"  # Update with actual image path
    result = validate_serial_number(image_path)
    print(result)
