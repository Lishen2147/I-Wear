import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def safe_pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except (OSError, SyntaxError):
        return Image.new('RGB', (224, 224))

def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    print("Starting training...")

    train_losses = []
    train_accuracies = []

    if device.type == 'cuda':
        scaler = GradScaler()
        # print("Using Scaler")
    else:
        scaler = None
        # print("Not using Scaler")

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                # print("Using Scaler")
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # print("Not using Scaler")
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_train_losses.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)

        accuracy = 100 * correct / total
        train_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {accuracy:.2f}%")

    return train_losses, train_accuracies

def test_model(model, test_loader, criterion, device):
    print("Starting testing...")

    model.eval()
    test_losses = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = np.mean(test_losses)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

def plot_and_save_training_history(train_losses, train_accuracies, save_path):
    # Plotting training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + "_loss.png")  # Save plot as an image
    plt.close()

    # Plotting training accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + "_accuracy.png")  # Save plot as an image
    plt.close()

    print("Plots saved successfully.")

def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)