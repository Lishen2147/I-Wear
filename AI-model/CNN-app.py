import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

def predict_face_shape(user_selfie_path, model_path, num_classes):
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
    predicted_index = torch.argmax(output, dim=1).item()
    face_shapes = ['Heart', 'Round', 'Oval', 'Square', 'Diamond', 'Oblong']  # Replace with your actual labels
    predicted_shape = face_shapes[predicted_index]

    return user_selfie, predicted_shape

# Example usage:
USER_SELFIE_PATH = './tests/test1.jpg'
MODEL_PATH = './outputs/best_model_kaggle.pth'
NUM_CLASSES = 5

user_selfie, predicted_shape = predict_face_shape(USER_SELFIE_PATH, MODEL_PATH, NUM_CLASSES)

print("Predicted face shape:", predicted_shape)