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

def character_recog_tesseract(img):
    # Resize the character image to improve OCR accuracy
    img = cv2.resize(img, (40, 40), interpolation=cv2.INTER_CUBIC)  # Resized for better clarity
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])  # Add thicker white border

    # Convert to grayscale and apply Gaussian blur for denoising
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform OCR on the preprocessed image
    custom_config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    # text = pytesseract.image_to_string(img_blur, config=custom_config)
    # Extract text with confidence > 50%
    data = pytesseract.image_to_data(img_blur, config=custom_config, output_type=pytesseract.Output.DICT)
    characters = [data['text'][i] for i in range(len(data['text']))]
    text = ''.join(characters)
    return text.strip()

def extract_text_from_image(image_path, Min_char=0.01, Max_char=0.09):
    # Load the image and apply initial processing
    # Read the image
    temp = cv2.imread(image_path)

    # Get the dimensions of the image
    height, width = temp.shape[:2]

    # Calculate the new width
    new_width = int(width * 1.3)

    # Resize the image
    image = cv2.resize(temp, (new_width, height))
    image_resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, rotate_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    LP_rotated_copy = image_resized.copy()

    # Find contours and filter them
    contours, _ = cv2.findContours(rotate_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:17]

    height, width, _ = LP_rotated_copy.shape
    roiarea = height * width
    char_x = []

    # Filter contours by area and aspect ratio
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

    # Sort characters by x-coordinate and identify line threshold
    char_x = sorted(char_x, key=lambda x: x[0])
    threshold_12line = min([c[1] for c in char_x]) + (np.mean([c[3] for c in char_x]) / 2)
    first_line, second_line = "", ""

    # Recognize characters and display
    for i, char in enumerate(char_x):
        x, y, w, h = char

        # Extract character region
        imgROI = rotate_thresh[y:y+h, x:x+w]

        # Resize the original image directly
        resized = cv2.resize(imgROI, (240, int(imgROI.shape[0] * (240 / imgROI.shape[1]))))

        # Show the resized image
        # cv2.imshow("a character" + str(i), resized)

        # Perform OCR
        text = character_recog_tesseract(cv2.cvtColor(imgROI, cv2.COLOR_GRAY2BGR))
        if text:  # Ignore empty predictions
            # Draw rectangle and put detected text
            cv2.rectangle(LP_rotated_copy, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.putText(LP_rotated_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

            # Sort characters into lines
            if y < threshold_12line:
                first_line += text
            else:
                second_line += text

    # Display the final image with detected characters
    resized_image = cv2.resize(LP_rotated_copy, (720, int(LP_rotated_copy.shape[0] * (720 / LP_rotated_copy.shape[1]))))
    cv2.imshow("Detected Characters", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Combine detected lines
    strFinalString = first_line + second_line
    return strFinalString

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
            orc_text = extract_text_from_image(object_path)
            print("This is orc text=======:",object_path, "=======:",orc_text)
            
# Cleanup
cv2.destroyAllWindows()
