import cv2
import time
import imutils

# Initialize the camera
cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None
area = 500
frame_counter = 0
frame_update_rate = 10  # Update the reference frame every 10 frames

while True:
    # Read frame from the camera
    _, img = cam.read()
    text = "Normal"
    img = imutils.resize(img, width=500)

    # Convert to grayscale and apply Gaussian Blur
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    # Capture the first frame or update the reference frame periodically
    if firstFrame is None or frame_counter % frame_update_rate == 0:
        firstFrame = gaussianImg
        frame_counter = 0

    # Compute the absolute difference
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)

    # Apply thresholding and dilate
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    # Find contours
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"

    # Print the detection status
    print(text)
    
    # Put text on the image
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the image
    cv2.imshow("cameraFeed", img)
    
    # Increment frame counter
    frame_counter += 1
    
    # Break the loop on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
