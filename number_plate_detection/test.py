import cv2
harcascade = "haarcascade_russian_plate_number.xml"
plate_detector = cv2.CascadeClassifier(harcascade)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(morph_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(frame, "Contour", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Edge", (x + w - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Edges and Contours", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()