import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import copy

def go_go_print(message=''):
    print('--------------------- ' + message + ' ---------------------')

def go_go_data_loaders (base_path = './flowers'):
    go_go_print('Loading data from base directory "%s"' % base_path)
    train_dir = base_path + '/train'
    valid_dir = base_path + '/valid'
    test_dir = base_path + '/test'

    # vars for transforms cause coders are lazy lol
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = transforms.Resize(255)
    center_crop = transforms.CenterCrop(224)

    # transformations
    data_transforms = {
        "training": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        "testing": transforms.Compose([
            resize,
            center_crop,
            transforms.ToTensor(),
            normalize,
        ]),
        "validation": transforms.Compose([
            resize,
            center_crop,
            transforms.ToTensor(),
            normalize,
        ])
    }

    # load out data sets
    image_datasets = {
        "training": datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        "testing": datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        "validation": datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }
    # load in our data loaders
    dataloaders = {
        "training": torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        "testing": torch.utils.data.DataLoader(image_datasets['testing'], batch_size=32, shuffle=True),
        "validation": torch.utils.data.DataLoader(image_datasets['validation'], batch_size=15, shuffle=True)
    }
    go_go_print('Data Loaded')
    return dataloaders, image_datasets

def go_go_categories():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def go_go_arch(arch):
    arch_types = {
        "densenet121":1024,
        "densenet169":1664,
        "vgg16":25088,
        "vgg11":25088,
        "alexnet":9216,
    }
    # Really dissapointed python doesn't have a switch statement
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        raise ValueError('Could not find architecture type "%s"' % arch)
    return model, arch_types[arch]


def go_go_model(arch='densenet121', hidden_layer = 512, num_of_categories = 102, lr=0.001, dropout=0.5, device='cuda'):
    go_go_print('Model Creation Init')
    model, architecture_size = go_go_arch(arch)
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    # update classifier
    model.classifier = nn.Sequential(OrderedDict([
                              ('dropout_1', nn.Dropout(p=dropout)),
                              ('c_input', nn.Linear(architecture_size, hidden_layer)),
                              ('relu_1', nn.ReLU()),
                              ('dropout_1', nn.Dropout(p=dropout)),
                              ('c_layer_2',nn.Linear(hidden_layer, hidden_layer)),
                              ('relu3',nn.ReLU()),
                              ('c_layer_3', nn.Linear(hidden_layer, num_of_categories)),
                              ]))
    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    # use cuda/gpu. if available
    if torch.cuda.is_available() and device == 'cuda':
        model.cuda()
    # return model, criterion, optimizer and model_settings (used for saving the model)
    go_go_print('Model Creation Completed')
    return model, criterion, optimizer,  [arch, hidden_layer, num_of_categories, lr, dropout, device]

def go_go_network(model, criterion, optimizer, dataloaders, datasets, epochs = 12, device='cuda'):
    go_go_print('Training Initialized with %s epochs' % epochs)
    steps = 0

    inputType = torch.device("cpu")
    if device == 'cuda' and torch.cuda.is_available():
        inputType = torch.device("cuda:0")
        model.cuda()

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for e in range(epochs):
        go_go_print('Epoch {}/{}'.format(e + 1, epochs))
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()  # training mode
            else:
                model.eval()   # evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(inputType)
                labels = labels.to(inputType)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}, {} Acc: {:.2f}'.format(
                phase, epoch_loss, phase, epoch_acc * 100))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

    # Load best model weights
    model.load_state_dict(best_model)

    # Store class_to_idx into a model property
    model.class_to_idx = datasets['training'].class_to_idx

    print('Best Accuracy: {:4f}'.format(best_acc))
    go_go_print('Training Completed')
    return model

def go_go_save_checkpoint(filepath, save_model, dataset, model_settings):
    save_model.class_to_idx = dataset.class_to_idx
    save_model.cpu
    torch.save({'state_dict': save_model.state_dict(),
            'arch': model_settings[0],
            'hidden_layers': model_settings[1],
            'num_of_categories': model_settings[2],
            'learning_rate': model_settings[3],
            'dropout': model_settings[4],
            'device': model_settings[5],
            'class_to_idx':save_model.class_to_idx},
            filepath)
    go_go_print('Checkpoint Saved')

def go_go_load_checkpoint(filepath='./checkpoint.pth'):
    checkpoint = torch.load(filepath)
    # hidden_layers = [1024, 300], num_of_categories = 102, lr=0.001, dropout=0.5, device='cuda'
    model, criterion, optimizer,  model_settings = go_go_model(
        checkpoint['arch'],
        checkpoint['hidden_layers'],
        checkpoint['num_of_categories'],
        checkpoint['learning_rate'],
        checkpoint['dropout'],
        checkpoint['device'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    go_go_print('Checkpoint Loaded')
    return model, criterion, optimizer,  model_settings

def go_go_process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_img = Image.open(image)
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transformations(pil_img)

def go_go_predict(image_path, model, topk=5, device='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img_torch = go_go_process_image(image_path)
    img_torch = img_torch.unsqueeze(0)
    if device == 'cuda' and torch.cuda.is_available():
        model.cuda()
        img_torch = img_torch.cuda()

    wasTraining = model.training
    model.eval()

    probabilityResults = model(img_torch).topk(topk)

    model.train(mode=wasTraining)

    if device == 'cuda' and torch.cuda.is_available():
        probabilities = F.softmax(probabilityResults[0].data, dim=1).cpu().numpy()[0]
        predictedCategories = probabilityResults[1].data.cpu().numpy()[0] + 1
    else:
        probabilities = F.softmax(probabilityResults[0].data, dim=1).numpy()[0]
        predictedCategories = probabilityResults[1].data.numpy()[0] + 1

    categories = go_go_categories()
    predictedCategories = [categories[str(x)] for x in predictedCategories]

    return probabilities, predictedCategories, categories

def go_go_check_accuracy(loader, model, device='cuda'):
    correct = 0
    total = 0
    iters = 0
    if torch.cuda.is_available() and device == 'cuda':
        model.to('cuda:0')
    with torch.no_grad():
        for images, labels in loader:
            if torch.cuda.is_available() and device == 'cuda':
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress = iters / len(loader) * 100
            print('Progress: %d' % progress)
            iters += 1
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
