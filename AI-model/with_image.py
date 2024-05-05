import cv2
import numpy as np

# Load an image
image_path = './tests/test21.jpg'
image = cv2.imread(image_path)

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=25, minSize=(30, 30))

# print(f"Number of faces detected: {len(faces)}")
assert len(faces) == 1, f"Only one face is allowed. Number of faces detected: {len(faces)}"

# Function to resize image to fit window size
def resize_to_fit_window(image):
    # Get dimensions of the window
    screen_width = 1080  # Change this to the width of your screen or window
    screen_height = 1080  # Change this to the height of your screen or window

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
    # Load the image in color
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if the image has an alpha channel. If not, add the alpha channel
    if img.shape[2] == 3:  # No alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Define a mask where white areas are detected
    # Adjust the upper threshold to fit the specific tone of white in your image
    lower = np.array([220, 220, 220, 255])
    upper = np.array([255, 255, 255, 255])
    white_mask = cv2.inRange(img, lower, upper)

    # Invert the mask to get black areas as white
    img[white_mask != 0] = [0, 0, 0, 0]  # Set fully transparent

    return img

# Define a function to overlay glasses
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


glasses_path = "./glasses"

# Define a dictionary to store the glasses options for each face shape
# Maps face predictions to possible glasses shapes
glasses_dict = {
    "heart": ["oval", "rectangle", "round"],
    "oblong": ["horn", "oval", "square"],
    "oval": ["oval", "rectangle", "round"],
    "round": ["oval", "rectangle", "square"],
    "square": ["large", "oval", "round"]
}

# Loop through the dictionary and prepend the glasses path
for key, value in glasses_dict.items():
    glasses_dict[key] = [f"{glasses_path}/{glasses}.png" for glasses in value]

faces_pred = "round"
glasses_idx = 0

overlays = []

def generate_overlays(image, faces, glasses_dict, faces_pred, glasses_idx):
    # Load glasses image
    glasses_image_path = glasses_dict[faces_pred][glasses_idx]
    glasses_image = remove_white_background(glasses_image_path)
    glasses_shape = glasses_image_path.split("/")[-1].split(".")[0]

    landmarks_list = []

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Detect facial landmarks (simplified example)
        landmarks = [(x + int(w * 0.35), y + int(h * 0.45)), (x + int(w * 0.65), y + int(h * 0.45))]
        landmarks_list.append(landmarks)
    
    for i in range(3):
        # Overlay the glasses onto the face
        output_image = overlay_glasses(image.copy(), glasses_image, landmarks_list)
        output_image = resize_to_fit_window(output_image)
        overlays.append(output_image)
        
        # # Display the resulting image
        # cv2.imshow(f'Glasses Overlay ({glasses_shape} glasses)', output_image)
    
        # # Wait for user input
        # key = cv2.waitKey(0)
        
        glasses_idx = (glasses_idx + 1) % len(glasses_dict[faces_pred])

    return overlays

generate_overlays(image, faces, glasses_dict, faces_pred, glasses_idx)

# while True:
#     # Load glasses image
#     glasses_image_path = glasses_dict[faces_pred][glasses_idx]
#     glasses_image = remove_white_background(glasses_image_path)
#     glasses_shape = glasses_image_path.split("/")[-1].split(".")[0]
    
#     landmarks_list = []

#     # Iterate over detected faces
#     for (x, y, w, h) in faces:
#         # Detect facial landmarks (simplified example)
#         landmarks = [(x + int(w * 0.35), y + int(h * 0.45)), (x + int(w * 0.65), y + int(h * 0.45))]
#         landmarks_list.append(landmarks)

#     # Overlay the glasses onto the face
#     output_image = overlay_glasses(image.copy(), glasses_image, landmarks_list)
#     output_image = resize_to_fit_window(output_image)

#     # Display the resulting image
#     cv2.imshow(f'Glasses Overlay ({glasses_shape} glasses)', output_image)
    
#     # Wait for user input
#     key = cv2.waitKey(0) & 0xFF

#     # If 'n' key is pressed, move to the next glasses option
#     if key == ord('n'):
#         glasses_idx = (glasses_idx + 1) % len(glasses_dict[faces_pred])

#     # If 'q' key is pressed, quit the loop
#     elif key == ord('q'):
#         break

# # Close all windows
# cv2.destroyAllWindows()
