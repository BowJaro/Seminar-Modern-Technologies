import cv2
import numpy as np
import os

def extract_characters_with_borders(image, output_folder, base_filename, min_char=0.01, max_char=0.09):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    height, width, _ = image.shape
    roi_area = height * width
    char_regions = []

    # Initial filter based on area and aspect ratio
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio_char = w / h
        char_area = w * h
        if (char_area > 1500) and (0.25 < ratio_char < 0.7):
            char_regions.append((x, y, w, h))

    # Filter by height consistency
    if char_regions:
        heights = [h for _, _, _, h in char_regions]
        median_height = np.median(heights)
        char_regions = [c for c in char_regions if abs(c[3] - median_height) < 0.1 * median_height]

    # Custom sorting logic: sort by y (row) and x (column) with tolerance
    def sort_key(region):
        x, y, w, h = region
        row_group = round(y / height)
        return (row_group, x)

    char_regions = sorted(char_regions, key=sort_key)

    # Create folders for saving bordered images and individual character images
    bordered_folder = os.path.join(output_folder, "green_borders")
    char_folder = os.path.join(output_folder, "extracted_characters", base_filename)
    os.makedirs(bordered_folder, exist_ok=True)
    os.makedirs(char_folder, exist_ok=True)

    # Draw borders and save character images
    for i, (x, y, w, h) in enumerate(char_regions):
        char_roi = thresh[y:y+h, x:x+w]

        # Save each character image
        char_path = os.path.join(char_folder, f"character{i+1}.jpg")
        cv2.imwrite(char_path, char_roi)

        # Draw green border around each character
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the bordered image
    bordered_path = os.path.join(bordered_folder, f"{base_filename}.jpg")
    cv2.imwrite(bordered_path, image)

def process_with_borders_and_extraction(final_plate_folder, output_folder):
    for filename in os.listdir(final_plate_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            plate_path = os.path.join(final_plate_folder, filename)
            plate_img = cv2.imread(plate_path)
            base_filename = os.path.splitext(filename)[0]

            # Draw green borders and extract characters
            extract_characters_with_borders(plate_img, output_folder, base_filename)

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
        # Resize the character image to the same size as the template
        resized_image = cv2.resize(image, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(resized_image, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)

        if score > max_score:
            max_score = score
            best_match = char

    return best_match if best_match else "?"

def read_characters_from_extracted_files(extracted_folder, template_folder):
    """Read characters from extracted character images using template matching."""
    templates = load_templates(template_folder)
    detected_texts = {}

    for plate_folder in os.listdir(extracted_folder):
        plate_path = os.path.join(extracted_folder, plate_folder)
        if os.path.isdir(plate_path):  # Only process directories
            detected_text = ""
            for char_file in sorted(
                os.listdir(plate_path), 
                key=lambda f: f.split("character")[1].split(".")[0].split('_')[0]
            ):
                char_path = os.path.join(plate_path, char_file)
                char_img = cv2.imread(char_path, cv2.IMREAD_GRAYSCALE)
                if char_img is None:
                    print(f"Warning: Failed to load character image '{char_file}'. Skipping.")
                    detected_text += "?"
                    continue

                best_match = match_template_to_character(char_img, templates)
                character =best_match.split('.')[0]
                if character == 'DD':
                    character = 'Đ'
                detected_text += character

            detected_texts[plate_folder] = detected_text

    return detected_texts

# Paths for the extracted characters and templates
template_folder = "templates/"
extracted_characters_folder = "result/extracted_characters/"
final_plate_folder = "result/final_plate/"
output_folder = "result/"

# Read characters from the extracted files
process_with_borders_and_extraction(final_plate_folder, output_folder)
detected_texts = read_characters_from_extracted_files(extracted_characters_folder, template_folder)

# Print all detected texts
for plate, text in detected_texts.items():
    print(f"{plate}: {text}")
