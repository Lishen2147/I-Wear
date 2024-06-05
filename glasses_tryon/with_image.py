import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
import os

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

def is_valid_image(img):
    # image_path = './AI-model/tests/test21.jpg'
    # image = cv2.imread(image_path)
    # Load the pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=25, minSize=(30, 30))
    return len(faces) == 1, faces
    # print(f"Number of faces detected: {len(faces)}")
    # assert len(faces) == 1, f"Only one face is allowed. Number of faces detected: {len(faces)}"


# Function to resize image to fit window size
def resize_to_fit_window(image):
    # Get dimensions of the window
    screen_width = 1080
    screen_height = 1080

    # Get dimensions of the image
    img_height, img_width = image.shape[:2]

    # Calculate scaling factors to fit image into window
    scale_x = screen_width / img_width
    scale_y = screen_height / img_height

    # Choose the minimum scaling factor to ensure the whole image fits into the window
    scale = min(scale_x, scale_y)

    # Resize the image
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)
    return resized_image

def remove_white_background(image_path):
    # Load the image in color with alpha channel
    img = cv2.imread("../glasses_tryon/" + image_path, cv2.IMREAD_UNCHANGED)
    # Check if the image has an alpha channel. If not, add the alpha channel
    if img.shape[2] == 3:  # No alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Define a mask where white areas are detected
    # Adjust the upper threshold to fit the specific tone of white in your image
    lower_white = np.array([220, 220, 220, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(img, lower_white, upper_white)

    # Set fully transparent pixels where white was detected
    img[np.where(white_mask == 255)] = [0, 0, 0, 0]
    return img

# Overlay glasses on the face
def overlay_glasses(face_img, glasses_img, landmarks):
    output_img = face_img.copy()
    for landmark in landmarks:
        # Extract eye positions from landmarks
        left_eye = landmark[0]
        right_eye = landmark[1]

        # Calculate the distance between the centers of the two eyes
        distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)

        # Calculate the scaling factor based on the eye distance and desired size
        scaling_factor = distance / glasses_img.shape[1] * 3.0

        # Resize the glasses image
        glasses_resized = cv2.resize(glasses_img, None, fx=scaling_factor, fy=scaling_factor)

        # Calculate the position to place the glasses
        glasses_x = int(left_eye[0] - glasses_resized.shape[1] * 0.33)
        glasses_y = int((left_eye[1] + right_eye[1]) / 2 - glasses_resized.shape[0] / 2)

        # Overlay the glasses onto the face image
        for i in range(glasses_resized.shape[0]):
            for j in range(glasses_resized.shape[1]):
                if glasses_resized[i, j, 2] != 0:
                    output_img[glasses_y + i, glasses_x + j, :] = glasses_resized[i, j, :3]
    return output_img

def generate_images(faces_pred, image_base64):
    glasses_path = "./glasses"
    glasses_dict = {
    "heart": ["oval", "rectangle", "round"],
    "oblong": ["horn", "oval", "square"],
    "oval": ["oval", "rectangle", "round"],
    "round": ["oval", "rectangle", "square"],
    "square": ["large", "oval", "round"]
    }
    decoded = base64.b64decode(image_base64)
    np_arr = np.frombuffer(decoded, np.uint8)
    
    # Decode the NumPy array to an image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    is_good_image, faces = is_valid_image(img)
    if not is_good_image:
        return []
    for key, value in glasses_dict.items():
        glasses_dict[key] = [f"{glasses_path}/{glasses}.png" for glasses in value]
    glasses_idx = 0
    overlays = []
    faces_pred = faces_pred.lower()
    for i in range(3):
        # Load the glasses image
        glasses_image_path = glasses_dict[faces_pred][glasses_idx]
        glasses_image = remove_white_background(glasses_image_path)
        glasses_shape = glasses_image_path.split("/")[-1].split(".")[0]
        landmarks_list = []
        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Detect facial landmarks (simplified example)
            landmarks = [(x + int(w * 0.35), y + int(h * 0.45)), (x + int(w * 0.65), y + int(h * 0.45))]
            landmarks_list.append(landmarks)
        # Overlay the glasses onto the face
        output_image = overlay_glasses(img.copy(), glasses_image, landmarks_list)
        output_image = resize_to_fit_window(output_image)
        overlays.append(output_image)
        
        glasses_idx = (glasses_idx + 1) % len(glasses_dict[faces_pred])
    # Display the overlays using matplotlib
    # fig, axes = plt.subplots(1, len(overlays), figsize=(12, 6))
    base64_images = []

    # Plot the overlayed images and get base64 strings
    for i, overlay in enumerate(overlays):
        # Convert overlay to RGB for displaying with matplotlib
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Display the overlay image
        # axes[i].imshow(overlay_rgb)
        # axes[i].axis('off')
        # glasses_shape = glasses_dict[faces_pred][i % len(glasses_dict[faces_pred])].split("/")[-1].split(".")[0]
        # axes[i].set_title(f'Glasses Overlay ({glasses_shape} glasses)')
        
        # Convert overlay image to base64
        base64_img = image_to_base64(overlay)
        base64_images.append(base64_img)
    return base64_images
    # plt.show()

    # Print the base64 strings
    # for idx, b64 in enumerate(base64_images):
        # print(f"Image {idx + 1} (Base64): {b64}")
