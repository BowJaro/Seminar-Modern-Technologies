import cv2
import os
import pytesseract
from pytesseract import Output

# Load Haar cascade for number plate detection (or object detection)
harcascade = "haarcascade_russian_plate_number.xml"
plate_detector = cv2.CascadeClassifier(harcascade)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

# Specify the folder containing the images
image_folder = 'data/'

# Create directories if not already existing
if not os.path.exists('result'):
    os.makedirs('result')

if not os.path.exists('result/extracted_objects'):
    os.makedirs('result/extracted_objects')

def draw_borders(image):
    # Process image to find contours but don't draw borders
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel, iterations=10)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=10)
    enhanced_contrast = cv2.add(gray, top_hat)
    enhanced_contrast = cv2.subtract(enhanced_contrast, black_hat)
    blurred = cv2.GaussianBlur(enhanced_contrast, (5, 5), 0)
    canny_edges = cv2.Canny(blurred, 100, 200)
    dilated_edges = cv2.dilate(canny_edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True)  # Calculate perimeter
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximate contour
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h  # Calculate the ratio of width to height
        
        # Continue finding valid contours based on aspect ratio and perimeter
        if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
            screenCnt.append(approx)

    return image  # Don't draw borders, just return the image as-is

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
        if area > 5000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
            [x, y, w, h] = cv2.boundingRect(approx.copy())
            ratio = w / h  # Calculate the ratio of width to height
            
            # Filter using ratio and vertices count
            if (len(approx) == 4) and (1.2 <= ratio <= 1.6 or 4.5 <= ratio <= 6.5):
                # Crop the object from the original image
                object_image = image[y:y + h, x:x + w]
                
                # Rename based on original image name
                object_path = f'result/extracted_objects/{image_name}_{object_count}.jpg'
                cv2.imwrite(object_path, object_image)
                object_paths.append(object_path)
                object_count += 1  # Increment the object count
    
    return object_paths

def perform_ocr_with_bounding_boxes(image_path, output_path):
    # Perform OCR and draw bounding boxes for each character
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, roi_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use Tesseract to get character data with bounding boxes
    config = "--dpi 300"  # High DPI for better results
    data = pytesseract.image_to_data(roi_thresh, output_type=Output.DICT, config=config)
    
    extracted_text = ""
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Only consider high-confidence characters
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            # Draw a green rectangle around each detected character
            cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            extracted_text += data['text'][i] + " "
    
    # Save the image with character-level bounding boxes
    cv2.imwrite(output_path, resized_image)
    return extracted_text.strip()

# Main execution flow to process images from the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        
        # Load the image
        image = cv2.imread(image_path)
        image_name, _ = os.path.splitext(filename)  # Get the name without extension
        
        # Extract objects from the image
        extracted_object_paths = extract_objects(image, image_name)
        
        # Perform OCR on each extracted object and draw character borders
        for object_path in extracted_object_paths:
            extracted_text = perform_ocr_with_bounding_boxes(object_path, f"result/processed_{image_name}_{object_path.split('/')[-1]}")
            print(f"Text extracted from {object_path}: {extracted_text}")
        
        # Process the image (draw green borders)
        processed_image = draw_borders(image)
        
        # Save the processed image with borders
        processed_image_path = f"result/processed_{filename}"
        cv2.imwrite(processed_image_path, processed_image)

# Cleanup
cv2.destroyAllWindows()
