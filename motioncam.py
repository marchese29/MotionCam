import numpy as np
import cv2, cv

DIFF_THRESHOLD = 30
MHI_DEPTH = 12
CAM_HEIGHT = 480
CAM_WIDTH = 640

cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

ret, frame = cap.read()
previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mhi = np.zeros_like(previous)

while True:
    ret, frame = cap.read()

    # Convert the frame to black and white
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find what's changed since the last frame.
    diff = np.zeros_like(gray)
    cv2.absdiff(gray, previous, diff)
    diff = (diff >= DIFF_THRESHOLD) * MHI_DEPTH

    # Build in the latest motion to the MHI
    mhi[mhi > 0] -= 1
    mhi = np.maximum(mhi, diff)

    # Set the current frame as the new previous
    previous = gray

    # Show the current state of the MHI
    cv2.imshow('frame', (mhi > 0) * gray)
    #cv2.imshow('frame', mhi.astype(float) / MHI_DEPTH)
    
    # Quit logic
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
