import Preprocess
import cv2
import os
import numpy as np
import math

# Load Haar cascade for number plate detection (or object detection)
harcascade = "haarcascade_russian_plate_number.xml"
plate_detector = cv2.CascadeClassifier(harcascade)

# Initialize camera
cap = cv2.VideoCapture(0)

# Create directories if not already existing
if not os.path.exists('result'):
    os.makedirs('result')

if not os.path.exists('result/extracted_objects'):
    os.makedirs('result/extracted_objects')

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

    # Process the image
    imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(captured_frame)
    cv2.imshow("Processed image", imgThreshplate)

    canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel,iterations=1)  # Dilation

    # Filter out license plates
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Pick out 5 biggest contours
    screenCnt = []

    print("Number of contours detected: ", len(contours))
    for c in contours:
        peri = cv2.arcLength(c, True)  # Tính chu vi
        # Draw the contour with a green line (BGR format)
        print("The perimeter is: ", peri)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximate the edges of contours
        cv2.drawContours(captured_frame, [approx], -1, (0, 255, 0), 3)
        print("The number of vertices is: ", len(approx))
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h
        if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
            screenCnt.append(approx)

    cv2.imshow("Contours", captured_frame)
    print("Number of screenCnt detected: ", len(screenCnt))
    if screenCnt is None:
        detected = 0
        print("No plate detected")
    else:
        detected = 1

    if detected == 1:
        n = 1
        for screenCnt in screenCnt:

            ################## Find the angle of the license plate ###############
            (x1, y1) = screenCnt[0, 0]
            (x2, y2) = screenCnt[1, 0]
            (x3, y3) = screenCnt[2, 0]
            (x4, y4) = screenCnt[3, 0]
            array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            sorted_array = array.sort(reverse=True, key=lambda x: x[1])
            (x1, y1) = array[0]
            (x2, y2) = array[1]

            doi = abs(y1 - y2)
            ke = abs(x1 - x2)
            angle = math.atan(doi / ke) * (180.0 / math.pi)
            #################################################

            # Masking the part other than the number plate
            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

            # Now crop
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = img[topx:bottomx + 1, topy:bottomy + 1]
            imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]

            ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

            if x1 < x2:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
            else:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

            roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
            imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)
            cv2.imshow("Line 116: Final processed image", imgThresh)

            # License Plate preprocessing
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
            cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Character segmentation
            char_x_ind = {}
            char_x = []
            height, width, _ = roi.shape
            roiarea = height * width
            # print ("roiarea",roiarea)
            for ind, cnt in enumerate(cont):
                area = cv2.contourArea(cnt)
                (x, y, w, h) = cv2.boundingRect(cont[ind])
                ratiochar = w / h
                if (Min_char * roiarea < area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                    if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                        x = x + 1
                    char_x.append(x)
                    char_x_ind[x] = ind

    
    # Wait for 'q' to quit or close the processed window
    if cv2.waitKey(0) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
