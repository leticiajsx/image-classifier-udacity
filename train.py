# Imports here
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict

# Example of how to run
# python train.py flowers --arch "densenet121" --learning_rate 0.01 --hidden_units 120 --epochs 8

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function for command line arguments
def parse_args_train():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint.")
    
    parser.add_argument('data_directory', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='/home/workspace/ImageClassifier', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture: vgg16, densenet121, or alexnet')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=120, help='Number of hidden units in the first layer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    return parser.parse_args()

# Function to set up the neural network architecture
def nn_setup(structure='vgg16', dropout=0.5, hidden_layer1=120, lr=0.001):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = model.classifier[0].in_features
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_features = model.classifier.in_features
    else:
        print("{} is not a valid model. Did you mean vgg16 or densenet121?".format(structure))
        return None, None

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(dropout)),
        ('inputs', nn.Linear(input_features, hidden_layer1)),
        ('relu1', nn.ReLU()),
        ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
        ('relu2', nn.ReLU()),
        ('hidden_layer2', nn.Linear(90, 80)),
        ('relu3', nn.ReLU()),
        ('hidden_layer3', nn.Linear(80, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device)

    return model, optimizer, criterion

def train():
    args = parse_args_train()

    # Load data
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets and dataloaders
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Configure the model
    model, optimizer, criterion = nn_setup(args.arch, 0.5, args.hidden_units, args.learning_rate)

    # Training
    print("Training started...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Print training statistics at each epoch
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {running_loss/len(train_loader):.4f}")

        # Validation on the validation set
        model.eval()
        validation_loss = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model.forward(inputs)
                validation_loss += criterion(outputs, labels).item()

        print(f"Validation Loss: {validation_loss/len(valid_loader):.4f}")

    # Evaluation on the test set after training
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)
            test_loss += criterion(outputs, labels).item()

    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

    # Save checkpoint
    model.class_to_idx = train_dataset.class_to_idx
    save_path = f"{args.save_dir}/checkpoint.pth"
    torch.save({
        'structure': args.arch,
        'hidden_layer1': args.hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }, save_path)

    print(f"Training completed. Model saved at {save_path}")

if __name__ == "__main__":
    train()