import cv2
import pytesseract
from pytesseract import Output
import os
from PIL import Image
import argparse

# Set tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Argument parser to handle image folder and preprocessing options
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pre_processor", default="thresh", help="Preprocessor to use: 'thresh' or 'blur'")
args = vars(ap.parse_args())

# Folder containing the images to process
image_folder = 'result1/'

# Main execution flow to process images from the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        
        # Load the image
        image = cv2.imread(image_path)
        image_name, _ = os.path.splitext(filename)  # Get the name without extension
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessor based on the argument passed
        if args["pre_processor"] == "thresh":
            print("here is thresh")
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif args["pre_processor"] == "blur":
            print("here is blur")
            gray = cv2.medianBlur(gray, 3)

        # Temporary filename to store the preprocessed image for pytesseract
        temp_filename = "{}.jpg".format(os.getpid())
        cv2.imwrite(temp_filename, gray)
        
        # Perform OCR on the processed image
        text = pytesseract.image_to_string(Image.open(temp_filename), lang='eng', 
                                           config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.')

        os.remove(temp_filename)  # Clean up temporary file
        
        # Output the image path and recognized text
        print(image_path, text)
