from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

# Construct the parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the pre-recorded video file")  # Argument for video file path
ap.add_argument("-a", "--min-area", type=int, default=2000, help="minimum area to parse")  # Argument for minimum area -- ,ale this represents the minimum area (in pixels) required for a contour to be considered as valid motion. By increasing this value, you can filter out smaller and less significant motions, which may help in reducing false positives in bright light situations.
args = vars(ap.parse_args())

# If the video argument is None, then we are reading from the camera
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()  # Start the video stream from the camera
    time.sleep(2.0)  # Allow the camera sensor to warm up
else:
    vs = cv2.VideoCapture(args["video"])  # Read video file

# Initialize the first frame in the video stream
firstFrame = None

while True:
    frame = vs.read()  # Read the next frame
    frame = frame if args.get("video", None) is None else frame[1]  # Extract frame from video capture result
    text = "Unoccupied"  # Initialize the text label for occupancy status

    if frame is None:
        break  # If there are no more frames, break from the loop

    frame = imutils.resize(frame, width=700)  # Resize the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    gray = cv2.GaussianBlur(gray, (85, 85), 0)  # Apply Gaussian blur to reduce noise - can change to a larger blur

    if firstFrame is None:
        firstFrame = gray
        continue  # Skip processing for the first frame

    frameDelta = cv2.absdiff(firstFrame, gray)  # Compute absolute difference between first frame and current frame
    thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]  # Threshold the frame delta image adjust this 25 parameters 50 or 100

    thresh = cv2.dilate(thresh, None, iterations=3)  # Dilate the thresholded image to fill in gaps
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the thresholded image
    cnts = imutils.grab_contours(cnts)  # Grab the appropriate contour list depending on OpenCV version

    for c in cnts:
        if cv2.contourArea(c) < args["min_area"] or len(c) < 5 or np.isinf(cv2.contourArea(c)):
            continue  # Ignore small or invalid contours

        (x, y, w, h) = cv2.boundingRect(c)  # Compute bounding box coordinates for the contour
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the bounding box rectangle on the frame
        text = "Intruders are here"  # Update the text label for occupancy status

        # Draw the text and the timestamp on the frame
        cv2.putText(frame, "Are there people?:  {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Show the frame and additional windows
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

# Cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
