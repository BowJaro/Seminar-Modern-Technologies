import cv2
import os
import numpy as np
import pytesseract
from pytesseract import Output


# Load Haar cascade for number plate detection (or object detection)
harcascade = "haarcascade_russian_plate_number.xml"
plate_detector = cv2.CascadeClassifier(harcascade)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

# Initialize camera
cap = cv2.VideoCapture(0)

# Create directories if not already existing
if not os.path.exists('result'):
    os.makedirs('result')

if not os.path.exists('result/extracted_objects'):
    os.makedirs('result/extracted_objects')

# def draw_borders(image):
#     """
#     Draw green borders around detected objects and return the processed image.
#     :param image: Input image where contours will be drawn.
#     :return: Processed image with borders.
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     cv2.imshow("gray_blurred", gray_blurred)
#     _, threshold = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     morph_opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    
#     contours, _ = cv2.findContours(morph_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 500:  # Adjust area threshold for sensitivity
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
    
#     return image

def draw_borders(image):
    """
    Draw green borders around detected objects (like number plates) and return the processed image.
    :param image: Input image where contours will be drawn.
    :return: Processed image with borders.
    """
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Increase contrast using a combination of TopHat and BlackHat morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel, iterations=10)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=10)
    enhanced_contrast = cv2.add(gray, top_hat)
    enhanced_contrast = cv2.subtract(enhanced_contrast, black_hat)

    # Step 3: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced_contrast, (5, 5), 0)

    # Step 4: Edge detection using Canny
    canny_edges = cv2.Canny(blurred, 100, 200)

    # Step 5: Dilation to enhance the edges
    dilated_edges = cv2.dilate(canny_edges, kernel, iterations=1)

    # Step 6: Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Filter contours based on area and aspect ratio (good for license plates)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust area threshold for sensitivity
            # Approximate contour to reduce points and smooth it
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.06 * peri, True)

            # Get bounding box and aspect ratio to filter possible license plates
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if (len(approx) == 4) and (0.8 <= aspect_ratio <= 1.5 or 4.5 <= aspect_ratio <= 6.5):
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle

    return image
    
def extract_objects(image):
    """
    Extract each object (contour) from the image and save it as a separate image.
    Ensures that only one object is extracted per image.
    :param image: Input image from which objects will be extracted.
    :return: List of extracted object image paths.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morph_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_paths = []
    object_count = 0
    
    # Iterate over all contours and ensure one object is saved per image
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust area threshold for sensitivity
            x, y, w, h = cv2.boundingRect(contour)
            
            # Crop the object from the original image
            object_image = image[y:y + h, x:x + w]
            
            # Save the extracted object image
            object_path = f'result/extracted_objects/object_{object_count}.jpg'
            if not os.path.exists('result/extracted_objects'):
                os.makedirs('result/extracted_objects')
            cv2.imwrite(object_path, object_image)
            object_paths.append(object_path)
            print(f"Extracted object saved at {object_path}")
            object_count += 1
    
    return object_paths

# def perform_ocr(image_path):
#     """
#     Perform OCR on the image and return the extracted text.
#     Applies pre-processing techniques such as resizing, grayscale conversion,
#     and denoising for better OCR accuracy.
    
#     :param image_path: Path to the image file.
#     :return: Extracted text from the image.
#     """
#     # Load the image
#     image = cv2.imread(image_path)
    
#     # Resize the image for better recognition (scaling by 2x)
#     resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
#     # Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply Otsu's thresholding after Gaussian filtering
#     _, roi_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Configure Tesseract OCR to recognize alphanumeric characters
#     # custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-'
    
#     # Perform OCR on the preprocessed image
#     # extracted_text = pytesseract.image_to_string(roi_thresh, config=custom_config)
#     extracted_text = pytesseract.image_to_string(image, lang='eng')
    
#     # Clean the extracted text
#     extracted_text = "".join(extracted_text.split()).replace(":", "").replace("-", "")
    
#     return extracted_text
def perform_ocr_with_bounding_boxes(image_path, output_path):
    """
    Perform OCR with bounding boxes drawn on the image and return the extracted text.
    This helps visualize the detected text regions.
    
    :param image_path: Path to the input image file.
    :param output_path: Path to save the output image with bounding boxes.
    :return: Extracted text from the image.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image for better recognition
    resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert the image to grayscale and denoise it
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to binarize the image
    _, roi_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform OCR and get detailed data (including bounding boxes and confidence)
    data = pytesseract.image_to_data(roi_thresh, output_type=Output.DICT)
    extracted_text = ""

    # Draw bounding boxes around detected text
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Confidence threshold for better accuracy
            x, y = data['left'][i], data['top'][i]
            w, h = data['width'][i], data['height'][i]
            
            # Draw rectangle on the original image
            cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Accumulate the text (filter out empty strings)
            extracted_text += data['text'][i] + " "

    # Save the image with bounding boxes for visualization
    cv2.imwrite(output_path, resized_image)

    # Clean up the extracted text
    extracted_text = extracted_text.replace(":", "").replace("-", "").strip()

    return extracted_text

def capture_image(cap):
    """
    Capture a frame from the webcam when 'c' is pressed.
    :param cap: The VideoCapture object.
    :return: Captured frame.
    """
    while True:
        ret, frame = cap.read()
        cv2.imshow("Press 'c' to capture, 'q' to quit", frame)
        
        key = cv2.waitKey(1)
        if key == ord('c'):  # Capture when 'c' is pressed
            return frame
        elif key == ord('q'):  # Quit the app when 'q' is pressed
            break

# Main execution flow
while True:
    captured_frame = capture_image(cap)
    
    if captured_frame is None:
        break
    
    # Save the captured image
    image_path = "result/captured_image.jpg"
    cv2.imwrite(image_path, captured_frame)
    print(f"Captured image saved at {image_path}")

    # Extract objects from the original captured frame (without borders)
    extracted_object_paths = extract_objects(captured_frame)
    
    # Perform OCR on each extracted object and print the text
    for object_path in extracted_object_paths:
        # extracted_text = perform_ocr(object_path)
        extracted_text = perform_ocr_with_bounding_boxes(object_path, "result/processed_image_with_borders.jpg")
        print(f"Text extracted from {object_path}: {extracted_text}")
    
    # Process the image (draw green borders)
    processed_image = draw_borders(captured_frame)
    
    # Save the processed image with borders
    processed_image_path = "result/processed_image_with_borders.jpg"
    cv2.imwrite(processed_image_path, processed_image)
    print(f"Processed image saved at {processed_image_path}")

    # Show the processed image
    cv2.imshow("Processed Image", processed_image)
    
    # Wait for 'q' to quit or close the processed window
    if cv2.waitKey(0) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
