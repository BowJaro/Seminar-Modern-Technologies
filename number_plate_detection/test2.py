import cv2
import pytesseract

harcascade = "haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)


min_area = 500


pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

while True:
    ret, frame = cap.read()

    plate_detector = cv2.CascadeClassifier(harcascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plates = plate_detector.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plate_img = frame[y:y + h, x:x + w]
            
            gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            gray_plate_img = cv2.GaussianBlur(gray_plate_img, (5, 5), 0)
            _, plate_img_thresh = cv2.threshold(gray_plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            plate_img_thresh = cv2.morphologyEx(plate_img_thresh, cv2.MORPH_OPEN, kernel)

            plate_text = pytesseract.image_to_string(plate_img_thresh, config='--psm 7')
            plate_text = "".join(c for c in plate_text if c.isalnum())
            
            roi_resized = cv2.resize(plate_img, (400, 100))
            cv2.imshow("ROI", roi_resized)
            frame = cv2.resize(frame, (512, 512))
            cv2.putText(frame, "Extracted Text: " + plate_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("PythonGeeks", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
