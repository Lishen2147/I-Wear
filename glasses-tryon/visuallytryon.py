# Adapted from https://github.com/AbdurRehmanMuhammad/Visually-Glasses-Try-on/tree/main

import cv2
import numpy as np

# Load the face cascade file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the glasses images with alpha channel
glasses_images = {
    'Glasses 1': cv2.imread('/glaa.png', cv2.IMREAD_UNCHANGED),
    # 'Glasses 2': cv2.imread('C://Users/manoo/Desktop/Visually Glasses Try-onsun1.png', cv2.IMREAD_UNCHANGED),
    # 'Glasses 3': cv2.imread('C://Users/manoo/Desktop/Visually Glasses Try-onglaa.png', cv2.IMREAD_UNCHANGED),
    # 'Glasses 4': cv2.imread('C://Users/manoo/Desktop/Visually Glasses Try-onglaa1.png', cv2.IMREAD_UNCHANGED),
}

# Function to overlay the glasses onto the face
def overlay_glasses(face_img, glasses_img, landmarks):
    output_img = face_img.copy()
    for landmark in landmarks:
        # Extract eye positions from landmarks
        left_eye = landmark[0]
        right_eye = landmark[1]

        # Calculate the distance between the centers of the two eyes
        distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)

        # Calculate the scaling factor based on the eye distance and desired size
        scaling_factor = distance / glasses_img.shape[1] * 3.

        # Resize the glasses image
        glasses_resized = cv2.resize(glasses_img, None, fx=scaling_factor, fy=scaling_factor)

        # Calculate the position to place the glasses
        glasses_x = int(left_eye[0] - glasses_resized.shape[1] * 0.33)
        glasses_y = int((left_eye[1] + right_eye[1]) / 2 - glasses_resized.shape[0] / 2)

        # Overlay the glasses onto the face image
        for i in range(glasses_resized.shape[0]):
            for j in range(glasses_resized.shape[1]):
                if glasses_resized[i, j, 3] != 0:
                    output_img[glasses_y + i, glasses_x + j, :] = glasses_resized[i, j, :3]

    return output_img

# Create a video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create an empty list to store the landmarks
    landmarks_list = []

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region of interest
        face_roi = frame[y:y+h, x:x+w]

        # Detect facial landmarks
        landmarks = [(x + int(w * 0.35), y + int(h * 0.45)), (x + int(w * 0.65), y + int(h * 0.45))]

        # Append the landmarks to the list
        landmarks_list.append(landmarks)

    # Get the selected glasses image
    glasses_choice = 'Glasses 1'  # Change this to select the desired glasses
    glasses_image = glasses_images[glasses_choice]

    # Overlay the glasses onto the face(s)
    output_image = overlay_glasses(frame.copy(), glasses_image, landmarks_list)

    # Display the resulting image
    cv2.imshow('Glasses Overlay', output_image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
