# Extract images by white pixel dominance
import cv2
import os
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

def create_directories(base_path):
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, 'first_extracted_images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'second_extracted_images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'gray'), exist_ok=True)
    os.makedirs('result/processed_images', exist_ok=True)

def extract_objects(image, image_name, output_folder, check_white=False):
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
        if area > 10000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
            [x, y, w, h] = cv2.boundingRect(approx.copy())
            ratio = w / h
            
            if (len(approx) == 4) and (1.2 <= ratio <= 1.8 or 3 <= ratio <= 7.5):
                object_image = image[y:y + h, x:x + w]
                
                if check_white:
                    if is_white_dominant(object_image):
                        object_path = f'{output_folder}/{image_name}_{object_count}.jpg'
                        cv2.imwrite(object_path, object_image)
                        object_paths.append(object_path)
                        object_count += 1
                else:
                    object_path = f'{output_folder}/{image_name}_{object_count}.jpg'
                    cv2.imwrite(object_path, object_image)
                    object_paths.append(object_path)
                    object_count += 1
    return object_paths

def is_white_dominant(cropped_image):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size
    white_pixels = np.sum(gray > 150)
    black_pixels = np.sum(gray < 90)
    return white_pixels > black_pixels

def process_image_for_black_white(image, output_path, delta=30):
    """
    Process an image to determine the two most popular colors, classify pixels into two groups,
    and generate a binary image with one group as black and the other as white.
    
    Args:
        image: The input image as a NumPy array (in BGR format).
        output_path: The path to save the processed binary image.
        delta: The threshold for pixel similarity to a color group.
    
    Returns:
        None. The processed image is saved to the specified output path.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat_pixels = gray.reshape(-1, 1)  # Flatten the grayscale image into a 1D array for clustering

    # Use KMeans to find the two dominant grayscale colors
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(flat_pixels)
    centers = np.sort(kmeans.cluster_centers_.flatten())  # Sort the cluster centers
    labels = kmeans.labels_

    # Extract the two dominant colors
    color_x, color_y = centers
    print(f"Detected colors: x={color_x:.2f}, y={color_y:.2f}")

    # Count pixels in the two groups
    group_x_count = np.sum(np.abs(flat_pixels - color_x) < delta)
    group_y_count = np.sum(np.abs(flat_pixels - color_y) < delta)

    print(f"Group X pixels: {group_x_count}, Group Y pixels: {group_y_count}")

    # Determine which color is closer to black (lower intensity)
    black_color = color_x if color_x < color_y else color_y

    # Create a binary image based on the classification
    binary_image = np.zeros_like(gray, dtype=np.uint8)
    binary_image[labels.reshape(gray.shape) == (0 if black_color == color_x else 1)] = 0  # Black group
    binary_image[labels.reshape(gray.shape) == (1 if black_color == color_x else 0)] = 255  # White group

    # Save the processed binary image
    cv2.imwrite(output_path, binary_image)
    print(f"Processed image saved to: {output_path}")

def process_images(image_folder, output_folder, check_white=False):
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            image_name, _ = os.path.splitext(filename)
            extract_objects(image, image_name, output_folder, check_white)


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

def main():
    image_folder = 'data/'
    result_base = 'result/'
    create_directories(result_base)

    count_correct = 0
    count_incorrect = 0
    
    first_extracted_folder = os.path.join(result_base, 'first_extracted_images')
    print("Starting first extraction...")
    process_images(image_folder, first_extracted_folder)
    
    # second_extracted_folder = os.path.join(result_base, 'second_extracted_images')
    second_extracted_folder = os.path.join(result_base, 'processed_images')
    print("Starting second extraction...")
    process_images(first_extracted_folder, second_extracted_folder, check_white=True)
    
    print("Extraction complete.")

    for filename in os.listdir(second_extracted_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(second_extracted_folder, filename)
            text = extract_and_compare_text(image_path)
            filename = filename.split('_')[0]
            is_equal = filename == text
            if is_equal:
                count_correct += 1
            else:
                count_incorrect += 1
            print(f"Extracted text from {filename.split('_')[0]}: {text} (Match: {is_equal})")
    
    print(f"Correct: {count_correct}, Incorrect: {count_incorrect}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
