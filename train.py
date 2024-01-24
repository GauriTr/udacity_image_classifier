import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn, optim
from torchvision import datasets, transforms, models

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="architecture", action="store", default="vgg16", type=str)
    parser.add_argument('--save_dir', dest="save_directory", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, type=float)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args

def transform_data(data_dir, transform_type):
    if transform_type == 'train':
        data_transforms = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    data = datasets.ImageFolder(data_dir, transform=data_transforms)
    return data

def create_data_loader(data, batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader

def set_device(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    
    return device

def load_pretrained_model(architecture="vgg16"):
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters():
        param.requires_grad = False

    return model

def modify_classifier(model, hidden_units):
    classifier = nn.Sequential(OrderedDict([
        ('inputs', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('hidden_layer1', nn.Linear(hidden_units, 90)),
        ('relu2', nn.ReLU()),
        ('hidden_layer2', nn.Linear(90, 70)),
        ('relu3', nn.ReLU()),
        ('hidden_layer3', nn.Linear(70, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return classifier

def validate_model(model, test_loader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def train_network(model, train_loader, valid_loader, device, criterion, optimizer, epochs, print_every):
    steps = 0
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validate_model(model, valid_loader, criterion, device)
                    
                print(f"Epoch: {e+1}/{epochs} | "
                      f"Training Loss: {running_loss/print_every:.4f} | "
                      f"Validation Loss: {valid_loss/len(valid_loader):.4f} | "
                      f"Validation Accuracy: {accuracy/len(valid_loader):.4f}")
                
                running_loss = 0
                model.train()

    return model

def test_accuracy(model, test_loader, device):
    correct, total = 0, 0
    
    with torch.no_grad():
        model.eval()
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test images: {accuracy:.2f}%')

def save_checkpoint(model, save_dir, train_data):
    if type(save_dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(save_dir):
            model.class_to_idx = train_data.class_to_idx
            
            checkpoint = {
                'architecture': model.name,
                'classifier': model.classifier,
                'class_to_idx': model.class_to_idx,
                'state_dict': model.state_dict()
            }
            
            torch.save(checkpoint, 'checkpoint.pth')
            print("Model checkpoint saved successfully.")
        else:
            print("Directory not found, model will not be saved.")

def main():
    # Get command line arguments for training
    args = parse_arguments()

    # Set directories for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transform and load data
    train_data = transform_data(train_dir, transform_type='train')
    valid_data = transform_data(valid_dir, transform_type='valid')
    test_data = transform_data(test_dir, transform_type='test')

    train_loader = create_data_loader(train_data, batch_size=50, shuffle=True)
    valid_loader = create_data_loader(valid_data, batch_size=50, shuffle=False)
    test_loader = create_data_loader(test_data, batch_size=50, shuffle=False)

    # Load pre-trained model
    model = load_pretrained_model(architecture=args.architecture)

    # Modify classifier
    model.classifier = modify_classifier(model, hidden_units=args.hidden_units)

    # Set device (CPU or GPU)
    device = set_device(gpu_arg=args.gpu)

    # Set learning rate
    learning_rate = args.learning_rate if args.learning_rate else 0.001

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Training hyperparameters
    print_every = 30

    # Train the network
    trained_model = train_network(model, train_loader, valid_loader, device, criterion, optimizer, args.epochs, print_every)

    print("\nTraining process is completed!")

    # Test accuracy on the test set
    test_accuracy(trained_model, test_loader, device)

    # Save model checkpoint
    save_checkpoint(trained_model, args.save_directory, train_data)

if __name__ == '__main__':
    main()
