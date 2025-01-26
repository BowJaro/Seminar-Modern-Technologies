import cv2
import os
import pytesseract
from pytesseract import Output
import numpy as np
from PIL import Image
import re
from collections import Counter

# Load Haar cascade for number plate detection (or object detection)
harcascade = "model/haarcascade_russian_plate_number.xml"
plate_detector = cv2.CascadeClassifier(harcascade)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

# Specify the folder containing the images
image_folder = 'data/'

# Create directories if not already existing
if not os.path.exists('result'):
    os.makedirs('result')

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

def character_recog_tesseract(img):
    # Resize and preprocess for better OCR accuracy
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])  # Add white border

    # Perform OCR on the processed image
    text = pytesseract.image_to_string(img,lang='eng', config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return text.strip()

def extract_text_from_image(image_path, Min_char=0.01, Max_char=0.09):
    # Load image and preprocess
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, rotate_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    LP_rotated_copy = image.copy()

    # Find contours and filter them
    contours, _ = cv2.findContours(rotate_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:17]

    height, width, _ = LP_rotated_copy.shape
    roiarea = height * width
    char_x = []

    # Initial filter by area and aspect ratio
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratiochar = w / h
        char_area = w * h
        if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
            char_x.append([x, y, w, h])

    # Further filter by height consistency (10% variance from median height)
    if char_x:
        heights = [h for _, _, _, h in char_x]
        median_height = np.median(heights)
        char_x = [c for c in char_x if abs(c[3] - median_height) < 0.1 * median_height]

    if not char_x:
        return ""  # Return empty string if no characters found

    # Sort characters by x-coordinate
    char_x = sorted(char_x, key=lambda x: x[0])
    threshold_12line = min([c[1] for c in char_x]) + (np.mean([c[3] for c in char_x]) / 2)
    first_line, second_line = "", ""

    # Recognize characters, draw borders, and display character labels
    for i, char in enumerate(char_x):
        x, y, w, h = char
        
        # Extract the character region
        imgROI = rotate_thresh[y:y+h, x:x+w]

        # Create a white background slightly larger than the character ROI
        white_background = np.ones((h + 10, w + 10), dtype=np.uint8) * 255
        
        # Place the character image in the center of the white background
        white_background[5:5+h, 5:5+w] = imgROI

        # Predict character using Tesseract
        text = character_recog_tesseract(white_background)
        if text:  # Ignore empty predictions
            # cv2.imshow("imgROI" + str(i), white_background)  # Show each character ROI with white background
            
            # Draw green border around each character
            cv2.rectangle(LP_rotated_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display detected character above the border
            cv2.putText(LP_rotated_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Place the text in either first or second line
            if y < threshold_12line:
                first_line += text
            else:
                second_line += text

    # Display the image with green borders and character labels
    # cv2.imshow("Detected Characters", LP_rotated_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Combine first and second lines
    strFinalString = first_line + second_line
    return strFinalString

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

def extract_and_compare_text(image_path):
    # Use both methods to extract text
    method1_text = extract_text_from_image(image_path)
    method2_text = process_single_image(image_path)

    len1 = len(method1_text)
    len2 = len(method2_text)

    different1 = abs(len1 - 8)
    different2 = abs(len2 - 8)

    if different1>= different2:
        return method2_text
    else:
        different1 = abs(len1 - 9)
        different2 = abs(len2 - 9)
        if different1>= different2:
            return method2_text
        else:
            return method1_text

# Count of true and false
count_true = 0
count_false = 0

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
            orc_text = extract_and_compare_text(object_path)
            if orc_text:
                file_name = object_path.split('/')[-1].split('_')[0] 
                result = file_name == orc_text
                if result:
                    count_true += 1
                else:
                    count_false += 1
                print(f"This is orc text======={object_path}======={orc_text}======={str(result):=>{100-len(object_path)-len(orc_text)-len('This is orc text=======:=======:=======:')}}")
print(f"True: {count_true}, False: {count_false}")

# Cleanup
cv2.destroyAllWindows()
