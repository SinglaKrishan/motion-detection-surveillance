import cv2
import imutils

# Step 1: Start webcam
cap = cv2.VideoCapture(0)
first_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 2: Resize for faster processing
    frame = imutils.resize(frame, width=500)

    # Step 3: Convert to grayscale & blur to remove noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Step 4: Store first frame as background
    if first_frame is None:
        first_frame = gray
        continue

    # Step 5: Find difference between current frame & first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Step 6: Dilate threshold image to fill gaps
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Step 7: Find contours (movement areas)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Ignore small movements
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Step 8: Show frames
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Frame Delta", frame_delta)

    # Step 9: Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
