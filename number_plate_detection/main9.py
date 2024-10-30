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
image_folder = 'clip4_new_15/'

# Create directories if not already existing
if not os.path.exists('result'):
    os.makedirs('result')
if not os.path.exists('result1'):
    os.makedirs('result1')
if not os.path.exists('temp'):
    os.makedirs('temp')

if not os.path.exists('result/extracted_objects'):
    os.makedirs('result/extracted_objects')

# def extract_objects(image, image_name):
#     # Extract objects (contours) from the image with enhanced precision
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, threshold = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     morph_opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
#     contours, _ = cv2.findContours(morph_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     object_paths = []
#     object_count = 1  # Start count from 1
    
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 10000:
#             peri = cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
#             [x, y, w, h] = cv2.boundingRect(approx.copy())
#             ratio = w / h  # Calculate the ratio of width to height
            
#             # Filter using ratio and vertices count
#             if (len(approx) == 4) and (1.2 <= ratio <= 1.6 or 4.5 <= ratio <= 6.5):
#                 # Crop the object from the original image
#                 object_image = image[y:y + h, x:x + w]
                
#                 # Rename based on original image name
#                 object_path = f'result/extracted_objects/{image_name}_{object_count}.jpg'
#                 cv2.imwrite(object_path, object_image)
#                 object_paths.append(object_path)
#                 object_count += 1  # Increment the object count
    
#     return object_paths

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
    temp = 0
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 10000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
            [x, y, w, h] = cv2.boundingRect(approx.copy())
            ratio = w / h  # Calculate the ratio of width to height
            
            # Draw green border around the contour on the temporary image
            temp_image = image.copy()
            cv2.drawContours(temp_image, [contour], -1, (0, 255, 0), 2)
            # Save the temporary image
            temp_image_path = f'temp/{image_name}_{temp}.jpg'
            cv2.imwrite(temp_image_path, temp_image)
            temp_image_path1 = f'temp/{image_name}_{temp}_process.jpg'
            cv2.imwrite(temp_image_path1, morph_opening)
            # print(f'{image_name}_{temp} '+f"Ratio: {ratio}, Approx: {len(approx)}" )
            temp = temp + 1
            
            
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

def process_single_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Temporary filename to store the preprocessed image for pytesseract
    temp_filename = "{}.jpg".format(os.getpid())
    cv2.imwrite(temp_filename, gray)
    cv2.waitKey(0)
    
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
            orc_text = process_single_image(object_path)
            print("This is orc text=======:",object_path, "=======:",orc_text)

# Cleanup
cv2.destroyAllWindows()
