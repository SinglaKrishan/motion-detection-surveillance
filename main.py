import cv2
import imutils
import os
import datetime
import smtplib
import ssl
from email.message import EmailMessage
import time

# --- Email Setup ---
EMAIL_SENDER = "your_email"
EMAIL_PASSWORD = "app_passowrd"  # from Gmail App Passwords
EMAIL_RECEIVER = "reciever_email"

def send_email_alert(image_path):
    subject = "⚠️ Motion Detected!"
    body = "Motion has been detected by your surveillance system. Snapshot attached."

    msg = EmailMessage()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach image
    with open(image_path, "rb") as f:
        file_data = f.read()
        file_name = f.name
    msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

    # Send mail
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)
        print(f"[ALERT] Email sent to {EMAIL_RECEIVER}")

output_dir = "captured_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Step 1: Start webcam
cap = cv2.VideoCapture(0)
first_frame = None
image_count=0
last_email_time = 0
last_saved_time = 0

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

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Ignore small movements
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if motion_detected:
        current_time = time.time()
        if current_time - last_saved_time > 5:  # Save ek image every 5 sec
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{output_dir}/motion_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Motion detected! Image saved: {filename}")
            last_saved_time = current_time

            # Email alert every 60 sec
            if current_time - last_email_time > 60:
                send_email_alert(filename)
                last_email_time = current_time
    
    # Step 8: Show frames
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Frame Delta", frame_delta)

    # Step 9: Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
