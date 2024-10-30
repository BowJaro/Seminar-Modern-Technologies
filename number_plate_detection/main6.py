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

# --- NEW CODE: Function to get skew angle ---
def getSkewAngle(cvImage) -> float:
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    # Adjust angle to be within -20 to 20 degrees
    print("this is angle: ", angle)
    while not (-20 < angle < 20):
        if angle < -20:
            angle += 90
        elif angle > 20:
            angle -= 90

    return -angle

# --- NEW CODE: Function to rotate image ---
# def rotateImage(cvImage, angle: float):
#     newImage = cvImage.copy()
#     (h, w) = newImage.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return newImage

def rotateImage(cvImage, angle: float):
    (h, w) = cvImage.shape[:2]
    center = (w // 2, h // 2)

    # Calculate the new bounding dimensions of the image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new dimensions of the image
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the actual rotation and return the image
    newImage = cv2.warpAffine(cvImage, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return newImage

# --- NEW CODE: Deskew function to be applied before OCR ---
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

def extract_objects(image, image_name):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(morph_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_paths = []
    object_count = 1
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
            [x, y, w, h] = cv2.boundingRect(approx.copy())
            ratio = w / h
            
            if (len(approx) == 4) and (1.2 <= ratio <= 1.6 or 4.5 <= ratio <= 6.5):
                object_image = image[y:y + h, x:x + w]
                object_path = f'result/extracted_objects/{image_name}_{object_count}.jpg'
                cv2.imwrite(object_path, object_image)
                object_paths.append(object_path)
                object_count += 1
    
    return object_paths


def process_single_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # --- Deskew the image first ---
    deskewed_image = deskew(image)
    cv2.imwrite("result1/"+image_path.split('/')[-1], deskewed_image)
    
    # List to store the OCR results for each rotation
    ocr_results = []
    
    # Rotate the deskewed image by 0, 90, 180, 270 degrees and perform OCR
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated_image = deskewed_image.copy()  # No need to rotate for 0 degrees
        else:
            rotated_image = rotateImage(deskewed_image, angle)  # Rotate image by the specified angle

        # Convert the rotated image to grayscale
        gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Temporary filename to store the preprocessed image for pytesseract
        temp_filename = "{}.jpg".format(os.getpid())
        cv2.imwrite(temp_filename, gray)
        
        # Perform OCR on the processed image
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(Image.open(temp_filename), lang='eng', config=config)

        os.remove(temp_filename)  # Clean up temporary file
        
        # Clean up the text and add it to the list
        cleaned_text = text.replace("\n", "").replace(' ', '')
        ocr_results.append(cleaned_text)
        
        # Print the text for each rotation
        print(f"OCR result for {angle}°: {cleaned_text}")
    
    # Return the longest text string from the different rotations
    longest_text = max(ocr_results, key=len)
    return longest_text
    
# Main execution flow to process images from the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        
        # Load the image
        image = cv2.imread(image_path)
        image_name, _ = os.path.splitext(filename)
        
        # Extract objects from the image
        extracted_object_paths = extract_objects(image, image_name)
        
        # Perform OCR on each extracted object
        for object_path in extracted_object_paths:
            name = object_path.split('/')[-1]
            orc_text = process_single_image(object_path)
            print("This is OCR text=======:", object_path, "=======:", orc_text)

# Cleanup
cv2.destroyAllWindows()
