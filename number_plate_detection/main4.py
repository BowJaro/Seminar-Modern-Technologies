import cv2
import os
import pytesseract
from pytesseract import Output
from PIL import Image
import numpy as np

# Load Haar cascade for number plate detection (or object detection)
harcascade = "model/haarcascade_russian_plate_number.xml"
plate_detector = cv2.CascadeClassifier(harcascade)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

# Specify the folder containing the images
image_folder = 'data/'

# Create directories if not already existing
if not os.path.exists('result'):
    os.makedirs('result')
if not os.path.exists('result1'):
    os.makedirs('result1')

if not os.path.exists('result/extracted_objects'):
    os.makedirs('result/extracted_objects')

def extract_objects(image, image_name):
    # Extract objects (contours) from the image with enhanced precision
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(morph_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_paths = []
    object_count = 1  # Start count from 1
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
            [x, y, w, h] = cv2.boundingRect(approx.copy())
            ratio = w / h  # Calculate the ratio of width to height
            
            # Filter using ratio and vertices count
            if (len(approx) == 4) and (1.2 <= ratio <= 1.8 or 3 <= ratio <= 7.5):
                # Crop the object from the original image
                object_image = image[y:y + h, x:x + w]
                
                # Rename based on original image name
                object_path = f'result/extracted_objects/{image_name}_{object_count}.jpg'
                cv2.imwrite(object_path, object_image)
                object_paths.append(object_path)
                object_count += 1  # Increment the object count
    
    return object_paths

def perform_ocr(image_path, output_path):
    # Perform OCR and draw bounding boxes (same as before)
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, roi_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite("result1/"+ output_path.split('/')[-1], roi_thresh)
    data = pytesseract.image_to_data(roi_thresh, output_type=Output.DICT)
    extracted_text = ""
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:
            x, y = data['left'][i], data['top'][i]
            w, h = data['width'][i], data['height'][i]
            extracted_text += data['text'][i] + " "
    result = extracted_text.strip()
    if len(result) != 0:
        cv2.imwrite(output_path, resized_image)
    return result

def process_single_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Temporary filename to store the preprocessed image for pytesseract
    temp_filename = "{}.jpg".format(os.getpid())
    cv2.imwrite(temp_filename, gray)
    
    # Perform OCR on the processed image
    config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(Image.open(temp_filename), lang='eng', config=config)

    os.remove(temp_filename)  # Clean up temporary file
    
    # Return the recognized text
    return text.replace("\n", "").replace(' ','')

# Main execution flow to process images from the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        
        # Load the image
        image = cv2.imread(image_path)
        image_name, _ = os.path.splitext(filename)  # Get the name without extension
        
        # Extract objects from the image
        extracted_object_paths = extract_objects(image, image_name)
        
        # Perform OCR on each extracted object
        for object_path in extracted_object_paths:
            name = object_path.split('/')[-1]
            extracted_text = perform_ocr(object_path, f"result/processed_{name}")
            orc_text = process_single_image(object_path)
            print("This is orc text=======:",object_path, "=======:",orc_text)
            # print(f"Text extracted from {object_path}: ||||||||||| {extracted_text}")

# Cleanup
cv2.destroyAllWindows()
