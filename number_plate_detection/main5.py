import cv2
import os
import pytesseract
from pytesseract import Output
import numpy as np
from PIL import Image
import re
from collections import Counter

# Specify the folder containing the images
image_folder = 'data/'

# Create directories if not already existing
output_folder = 'result/'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, 'extracted_objects'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'extracted_characters'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'green_borders'), exist_ok=True)

def load_templates(template_folder):
    """Load character templates from a folder."""
    templates = {}
    for filename in os.listdir(template_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            char = os.path.splitext(filename)[0]  # Extract character name (e.g., '0', 'A')
            template_path = os.path.join(template_folder, filename)
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

            if template_img is None:
                print(f"Warning: Failed to load template for '{filename}'. Skipping.")
                continue

            _, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            templates[char] = template_img

    if not templates:
        raise ValueError("No valid templates loaded. Please check the template folder.")

    return templates

def match_template_to_character(image, templates):
    """Match a single character image to the templates and return the best match."""
    if image is None:
        print("Warning: Character image is None. Skipping matching.")
        return "?"

    best_match = None
    max_score = -1

    for char, template in templates.items():
        resized_image = cv2.resize(image, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(resized_image, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)

        if score > max_score:
            max_score = score
            best_match = char

    return best_match if best_match else "?"

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
        if area > 10000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
            [x, y, w, h] = cv2.boundingRect(approx.copy())
            ratio = w / h

            if (len(approx) == 4) and (1.2 <= ratio <= 1.8 or 3 <= ratio <= 7.5):
                object_image = image[y:y + h, x:x + w]
                object_path = f'{output_folder}/extracted_objects/{image_name}_{object_count}.jpg'
                cv2.imwrite(object_path, object_image)
                object_paths.append(object_path)
                object_count += 1

    return object_paths

def process_images(image_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            image_name, _ = os.path.splitext(filename)

            extract_objects(image, image_name)

def extract_characters_with_borders(image, output_folder, base_filename, templates):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    height, width, _ = image.shape
    roi_area = height * width
    char_regions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio_char = w / h
        char_area = w * h
        if (char_area > 1500) and (0.25 < ratio_char < 0.7):
            char_regions.append((x, y, w, h))

    if char_regions:
        heights = [h for _, _, _, h in char_regions]
        median_height = np.median(heights)
        char_regions = [c for c in char_regions if abs(c[3] - median_height) < 0.1 * median_height]

    char_regions = sorted(char_regions, key=lambda c: (round(c[1] / height), c[0]))

    char_folder = os.path.join(output_folder, "extracted_characters", base_filename)
    os.makedirs(char_folder, exist_ok=True)

    detected_text = ""

    for i, (x, y, w, h) in enumerate(char_regions):
        char_roi = thresh[y:y+h, x:x+w]
        char_path = os.path.join(char_folder, f"character{i+1}.jpg")
        cv2.imwrite(char_path, char_roi)

        best_match = match_template_to_character(char_roi, templates)
        detected_text += best_match.split(".")[0]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    bordered_path = os.path.join(output_folder, "green_borders", f"{base_filename}.jpg")
    cv2.imwrite(bordered_path, image)

    return detected_text

def read_characters(image_folder, output_folder, template_folder):
    templates = load_templates(template_folder)
    detected_texts = {}
    # Count of true and false
    count_true = 0
    count_false = 0

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        plate_img = cv2.imread(image_path)
        detected_text = extract_characters_with_borders(plate_img, output_folder, image_name, templates)
        detected_texts[image_name] = detected_text
        
        result = image_name.split('_')[0] == detected_text
        if result:
            count_true += 1
        else:
            count_false += 1
        print(f"This is orc text======={image_name}======={detected_text}{str(result):=>{75-len(image_name)-len(detected_text)-len('This is orc text=======:=======:=======:')}}")

    print(f"True: {count_true}, False: {count_false}")
    return detected_texts

# Paths for templates and images
template_folder = "templates/"
print("Extracting images...")
process_images(image_folder)

print("Reading characters...")
read_characters("result/extracted_objects", output_folder, template_folder)

# Cleanup
cv2.destroyAllWindows()
