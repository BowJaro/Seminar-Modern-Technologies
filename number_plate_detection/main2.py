# rotate number plate image to the correct orientation
import cv2
import os
import numpy as np
import math
import pytesseract
from pytesseract import Output
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def calculate_rotation_angle(approx, image_name, object_count):
    # Sort points by y value to find the lowest edge
    approx_sorted = sorted(approx[:, 0], key=lambda x: x[1])
    approx = np.array(approx_sorted)

    # Take the two lowest points to form the lowest edge
    pt1, pt2 = approx[2], approx[3]
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    angle = math.degrees(math.atan2(dy, dx))

    # Adjust angle based on conditions
    if -180 < angle < -135:
        angle += 180
    elif 135 < angle < 180:
        angle -= 180

    # Visualize angle
    visualize_angle(approx, (pt1, pt2), angle, image_name, object_count)
    return angle


def visualize_angle(approx, edge_pts, angle, image_name, object_count):
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    scale_factor = 400 / max(approx[:, 1].max(), approx[:, 0].max())
    scaled_approx = (approx * scale_factor).astype(int)

    cv2.polylines(canvas, [scaled_approx], isClosed=True, color=(0, 255, 0), thickness=2)
    pt1, pt2 = [tuple((pt * scale_factor).astype(int)) for pt in edge_pts]
    cv2.line(canvas, pt1, pt2, (0, 0, 255), 2)

    text = f"Angle: {angle:.2f} degrees"
    cv2.putText(canvas, text, (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    os.makedirs('result/angle_visualizations', exist_ok=True)
    visualization_path = f'result/angle_visualizations/{image_name}_angle_visualization_{object_count}.jpg'
    cv2.imwrite(visualization_path, canvas)


def rotate_image_and_save(image, angle, image_name, object_count):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Calculate the new bounding dimensions after rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])

    # Compute new width and height of the rotated image
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix to account for the translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Create a white canvas with the new dimensions
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # White background
    )

    # Save the rotated image
    os.makedirs('result/rotated_images', exist_ok=True)
    rotated_image_path = f'result/rotated_images/{image_name}_rotated_{object_count}.jpg'
    cv2.imwrite(rotated_image_path, rotated_image)

    return rotated_image


def extract_objects(image, image_name):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(morph_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    process_contours(image, contours, image_name)


def extract_final_plate(rotated_image, image_name, object_count):
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(morph_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            approx = cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                ratio = w / h
                if 0.8 <= ratio <= 1.8 or 3 <= ratio <= 7.5:
                    final_plate_region = rotated_image[y:y+h, x:x+w]
                    os.makedirs('result/final_plate', exist_ok=True)
                    final_plate_path = f'result/final_plate/{image_name}_final_{object_count}.jpg'
                    cv2.imwrite(final_plate_path, final_plate_region)
                    return final_plate_region

    # Fallback: Return the whole rotated image if no plate is found
    os.makedirs('result/final_plate', exist_ok=True)
    fallback_path = f'result/final_plate/{image_name}_final_fallback_{object_count}.jpg'
    cv2.imwrite(fallback_path, rotated_image)
    return rotated_image


def process_contours(image, contours, image_name):
    object_count = 1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            approx = cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                ratio = w / h
                if 0.8 <= ratio <= 1.8 or 3 <= ratio <= 7.5:
                    # Draw green border for visualization
                    bordered_image = image.copy()
                    cv2.polylines(bordered_image, [approx], isClosed=True, color=(0, 255, 0), thickness=3)
                    os.makedirs('result/bordered_images', exist_ok=True)
                    bordered_path = f'result/bordered_images/{image_name}_bordered_{object_count}.jpg'
                    cv2.imwrite(bordered_path, bordered_image)

                    # Crop the license plate region
                    x, y, w, h = cv2.boundingRect(approx)
                    license_plate_region = image[y:y+h, x:x+w]
                    os.makedirs('result/license_plate_regions', exist_ok=True)
                    cropped_path = f'result/license_plate_regions/{image_name}_cropped_{object_count}.jpg'
                    cv2.imwrite(cropped_path, license_plate_region)

                    # Calculate rotation angle and visualize
                    angle = calculate_rotation_angle(approx, image_name, object_count)

                    # Rotate and save the cropped license plate image
                    rotated_image = rotate_image_and_save(license_plate_region, angle, image_name, object_count)

                    # Extract the final number plate from the rotated image
                    extract_final_plate(rotated_image, image_name, object_count)

                    object_count += 1


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

# Main execution flow
image_folder = 'data/'
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        image_name, _ = os.path.splitext(filename)

        # Process image to detect and process contours
        extract_objects(image, image_name)

        
final_plate_folder = 'result/final_plate/'
for object_path in os.listdir(final_plate_folder):
    orc_text = extract_and_compare_text(final_plate_folder + object_path)
    if orc_text:
        print("This is orc text=======:",object_path, "=======:",orc_text)
        
cv2.destroyAllWindows()

