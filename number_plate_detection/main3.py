import cv2
import os
import numpy as np

# Set the folder path for templates
template_folder = 'templates/'

def load_templates(template_folder):
    """Load template images from a folder."""
    templates = {}
    for file in os.listdir(template_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            label = os.path.splitext(file)[0].upper()
            template = cv2.imread(os.path.join(template_folder, file), cv2.IMREAD_GRAYSCALE)
            templates[label] = template
    return templates

def match_number_plate_template(image, templates):
    """Match the number plate region using template matching."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best_match = None
    best_score = float('inf')
    match_rect = None

    for label, template in templates.items():
        result = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)

        if min_val < best_score:
            best_score = min_val
            best_match = label
            match_rect = (min_loc[0], min_loc[1], template.shape[1], template.shape[0])

    return match_rect

def extract_number_plate_region(image_path, templates, output_folder):
    """Extract and save the number plate region using template matching."""
    image = cv2.imread(image_path)
    match_rect = match_number_plate_template(image, templates)

    if match_rect:
        x, y, w, h = match_rect
        number_plate_region = image[y:y+h, x:x+w]

        # Save the extracted region
        os.makedirs(output_folder, exist_ok=True)
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"number_plate_{base_name}")
        cv2.imwrite(output_path, number_plate_region)

        print(f"Number plate region saved: {output_path}")

def main():
    """Main function to process images and extract number plates."""
    image_folder = 'data/'  # Input image folder
    output_folder = 'result/number_plate_regions/'  # Output folder for extracted regions

    # Load number plate templates
    templates = load_templates(template_folder)

    # Process each image in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)

            # Extract and save the number plate region
            extract_number_plate_region(image_path, templates, output_folder)

if __name__ == "__main__":
    main()
