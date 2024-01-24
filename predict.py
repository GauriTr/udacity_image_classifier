import argparse
import json
import PIL.Image
import torch
import numpy as np
from math import ceil
from train import check_gpu
from torchvision import models

def parse_arguments():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Path to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    return args

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model = models.vgg16(pretrained=True)
    model.name = "vgg16"

    for param in model.parameters():
        param.requires_grad = False

    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image_path):
    img = PIL.Image.open(image_path)

    original_width, original_height = img.size

    if original_width < original_height:
        size = [256, 256**600]
    else:
        size = [256**600, 256]

    img.thumbnail(size)

    center = original_width / 4, original_height / 4
    left, top, right, bottom = center[0] - (244 / 2), center[1] - (244 / 2), center[0] + (244 / 2), center[1] + (244 / 2)
    img = img.crop((left, top, right, bottom))

    numpy_image = np.array(img) / 255

    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalized_image = (numpy_image - mean) / std

    # Set the color to the first channel
    transposed_image = normalized_image.transpose(2, 0, 1)

    return torch.from_numpy(transposed_image).unsqueeze(0)

def predict(image_tensor, model, device, cat_to_name, topk=5):
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    probabilities = torch.exp(output)
    top_probabilities, top_indices = probabilities.topk(topk)
    
    top_probabilities = top_probabilities.cpu().numpy().squeeze()
    top_indices = top_indices.cpu().numpy().squeeze()

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_labels = [idx_to_class[index] for index in top_indices]
    top_flowers = [cat_to_name[label] for label in top_labels]

    return top_probabilities, top_labels, top_flowers

def print_probabilities(probs, flowers):
    for i, (flower, prob) in enumerate(zip(flowers, probs)):
        print("Rank {}: Flower: {}, Likelihood: {:.2f}%".format(i + 1, flower, prob * 100))

def main():
    args = parse_arguments()

    with open(args.category_names, 'r') as file:
        cat_to_name = json.load(file)

    model = load_checkpoint(args.checkpoint)
    image_tensor = process_image(args.image)
    device = check_gpu(gpu_arg=args.gpu)

    top_probabilities, top_labels, top_flowers = predict(image_tensor, model, device, cat_to_name, args.top_k)
    
    print_probabilities(top_probabilities, top_flowers)

if __name__ == '__main__':
    main()
