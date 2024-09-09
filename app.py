import cv2

# Path to Haar cascade and video
haar_cascade = 'cars.xml'
video = 'video.mp4'

# Open video capture
cap = cv2.VideoCapture(video)
car_cascade = cv2.CascadeClassifier(haar_cascade)

# Loop runs if capturing has been initialized.
while True:
    # Read frames from video
    ret, frames = cap.read()
    if not ret:
        break  # Break the loop if video has ended
    
    # Convert to gray scale of each frame
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Draw rectangle and label "Car" for each detected vehicle
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw rectangle
        cv2.putText(frames, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # Add label above rectangle

    # Display frames in a window 
    cv2.imshow('video', frames)

    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
