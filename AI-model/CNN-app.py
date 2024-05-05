import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

def predict_face_shape(user_selfie_path, model_path, num_classes):
    """
    Predicts the face shape of a user's selfie image using a pre-trained convolutional neural network model.

    Args:
        user_selfie_path (str): The file path of the user's selfie image.
        model_path (str): The file path of the pre-trained model.
        num_classes (int): The number of classes or face shapes to predict.

    Returns:
        tuple: A tuple containing the user's selfie image and a list of predicted face shapes with their probabilities.
            - user_selfie (PIL.Image.Image): The user's selfie image.
            - predicted_shapes_with_probabilities (list): A list of tuples containing the predicted face shape and its probability.
                Each tuple has the format (face_shape, probability).

    """

    # Load the model
    model = torchvision.models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define preprocessing transformations
    preprocess = T.Compose([
        T.Resize((224, 224)),  # Resize to match model input size
        T.ToTensor(),           # Convert to tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load and preprocess the user's selfie image
    user_selfie = Image.open(user_selfie_path)
    input_tensor = preprocess(user_selfie)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Decode predictions
    probabilities = torch.softmax(output, dim=1)[0]  # Convert logits to probabilities
    face_shapes = ['Heart', 'Round', 'Oval', 'Square', 'Diamond', 'Oblong']  # Replace with your actual labels
    predicted_shapes_with_probabilities = [(face_shape, probability.item()) for face_shape, probability in zip(face_shapes, probabilities)]

    return user_selfie, predicted_shapes_with_probabilities

# Example usage:
USER_SELFIE_PATH = './tests/test1.jpg'
MODEL_PATH = './outputs/best_model_kaggle.pth'
NUM_CLASSES = 5

user_selfie, predicted_shapes_with_probabilities = predict_face_shape(USER_SELFIE_PATH, MODEL_PATH, NUM_CLASSES)

print("Predicted face shapes with probabilities:")
for shape, probability in predicted_shapes_with_probabilities:
    print(f"Face shape: {shape}, Probability: {probability:.2f}")