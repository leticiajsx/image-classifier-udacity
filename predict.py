import argparse
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import json

# Example of how to run
# python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to parse command line arguments
def parse_args_predict():
    parser = argparse.ArgumentParser(description="Predict flower name from an image and the probability of that name.")
    
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    return parser.parse_args()

# Function to load the model from a checkpoint
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = model.classifier[0].in_features
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_features = model.classifier.in_features
    else:
        print("{} is not a valid model. Did you mean vgg16 or densenet121?".format(structure))
        return None

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(input_features, hidden_layer1),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_layer1, 90),
        nn.ReLU(),
        nn.Linear(90, 80),
        nn.ReLU(),
        nn.Linear(80, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']

    return model

# Function to process the input image
def process_image(image_path):
    img = Image.open(image_path)

    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Process the image
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

# Function to make predictions
def predict():
    args = parse_args_predict()

    # Check if GPU is requested and available
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the model
    model = load_model(args.checkpoint)
    model.to(device)
    model.eval()

    # Process the input image
    image_tensor = process_image(args.input)
    image_tensor = image_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    # Calculate probabilities and classes
    probabilities = torch.exp(output)
    top_probs, top_classes = probabilities.topk(args.top_k)

    # Convert indices to class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[idx.item()] for idx in top_classes[0]]

    # Load the category names mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Map class labels to flower names
    top_names = [cat_to_name[label] for label in top_labels]

    # Print the results
    print(f"\nTop {args.top_k} predictions for the flower in '{args.input}':")
    for i in range(args.top_k):
        print(f"{i + 1}. {top_names[i]} with probability {top_probs[0][i]:.4f}")

if __name__ == "__main__":
    predict()