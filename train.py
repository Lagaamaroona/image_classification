# Import libraries
import io
import json
import torch
import argparse
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from io import BytesIO
from torch import optim
from collections import OrderedDict

def data_preparation(data_container, training_batch, validation_testing_batch):
    #TODO: validate if the directory
    data_dir = data_container
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    transforms_training = transforms.Compose([
                                            transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))
                                            ])

    transforms_validation = transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),
                                                                    (0.229, 0.224, 0.225))
                                            ])

    transforms_testing = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))
                                            ])


    # Load the datasets with ImageFolder
    dataset_training = torchvision.datasets.ImageFolder(train_dir, transform = transforms_training)
    dataset_validation = torchvision.datasets.ImageFolder(valid_dir, transform = transforms_validation)
    dataset_testing = torchvision.datasets.ImageFolder(test_dir, transform = transforms_testing)

    # Using the image datasets and the trainforms, define the dataloaders
    dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=training_batch, shuffle=True, drop_last=True)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=validation_testing_batch)
    dataloader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=validation_testing_batch)

    return dataloader_training, dataloader_validation, dataloader_testing, dataset_training



def download_pretrained_model(model_name):
    model = None
    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    if model_name == "vgg19":
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    return model


def define_classifier(input_size, hidden_sizes, output_size):
    # TODO: Make it configurable based on the hidden_sizes list size
    classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_sizes[0])),
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(hidden_sizes[1], output_size)),
            ('output', nn.LogSoftmax(dim=1))]))
    return classifier


def validation(model, dataloader_validation, criterion, validation_testing_batch, gpu):
    valid_correct = 0
    valid_loss = 0

    for images, labels in (iter(dataloader_validation)):
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')

        # Forward pass
        output = model.forward(images)
        loss = criterion(output, labels)

        # Track loss
        valid_loss += loss.item()
        _, output_v = torch.max(output.data, 1)

        # Track Accuracy
        valid_correct += (output_v == labels).sum()
    
    return valid_loss/len(dataloader_validation), (100*valid_correct)/(validation_testing_batch*len(dataloader_validation))



def training(epochs, training_batch, validation_testing_batch, gpu):
    print("Training started....")
    for e in range(epochs):
        start = datetime.datetime.now()
        
        training_loss = 0
        training_corr = 0
        steps = 0
        print_every = 30

        for images, labels in iter(dataloader_training):
            # Setup
            steps += 1
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images)       
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Track loss
            ## loss contains loss value and the grad_fn, when loss.item() contains only the value.
            training_loss += loss.item() 

            # Track accuracy
            ## output contains the output and the grad_fn, when output.data contains only the output.
            _, output_t = torch.max(output.data, 1)
            training_corr += (output_t == labels).sum()

            if(steps % print_every == 0):
                model.eval()
                
                with torch.no_grad():
                    validation_loss, validation_accu = validation(model, dataloader_validation, criterion, validation_testing_batch, gpu)
                    print("Epoch:{}/{} ".format(e + 1, epochs),
                        "Training Loss: {:.3f} ".format(training_loss/print_every),
                        "Training Accu: {:.3f}% ".format((100*training_corr)/(print_every*training_batch)),
                        "Validation Loss: {:.3f} ".format(validation_loss),
                        "Validation Accu: {:.3f}%".format(validation_accu))

                training_loss = 0
                training_corr = 0
                
                model.train()

    end = datetime.datetime.now()
    total_duration = end - start
    print('Total time spent in this epoch to train the model: {} mins'.format(total_duration.total_seconds()/60))



def testing(model, dataloader_testing, criterion, validation_testing_batch, gpu):
    print("Testing started...")
    
    testing_correct = 0
    testing_loss = 0
    
    start = datetime.datetime.now()
    model.eval()
    
    for images, labels in (iter(dataloader_testing)):
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')

        # Forward pass
        output = model.forward(images)
        loss = criterion(output, labels)

        # Track loss
        testing_loss += loss.item()
        _, output_t = torch.max(output.data, 1)

        # Track Accuracy
        testing_correct += (output_t == labels).sum()

    model.train()
    
    end = datetime.datetime.now()
    total_duration = end - start

    testing_loss_percentage = testing_loss/len(dataloader_testing)
    testing_accu_percentage = (100*testing_correct)/(validation_testing_batch*len(dataloader_testing))
    
    print("Testing Loss: {:.3f} ".format(testing_loss_percentage), "Testing Accu: {:.3f}%".format(testing_accu_percentage))
    print('Total time spent to test the model: {} mins'.format(total_duration.total_seconds()/60))


def save_checkpoint(model, model_name, criterion, optimizer, dataset_training, epochs, saving_path):
    savedata = {
            'model_state': model.state_dict(),
            'criterion_state': criterion.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'class_to_idx': dataset_training.class_to_idx,
            'epoch': epochs,
            'model_name': model_name
           }

    torch.save(savedata, saving_path + 'trained_model.pth')
    print("trained model has been saved to " + saving_path + "trained_model.pth")
    print("application closing...")

if __name__ == "__main__":

    ### Default values ###
    # args.learning_rate = 0.001
    # args.gpu = True
    # args.epochs = 20
    # args.save_dir = "./"
    # args.arch = "None"

    ### Required inputs ###
    # args.hidden_units : list values i.e 1024 256

    parser = argparse.ArgumentParser(description="Train a predictive model using images")
    # Required
    parser.add_argument("-d", "--data_container", type=str, required=True, help="location of data to utilize")
    parser.add_argument("-u", "--hidden_units", type=int, nargs='+', required=True, help="number of hidden units in the neural network")
    # Optional
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate of the optimizer")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of iterations to perform during the training")
    parser.add_argument("-g", "--gpu", help="use GPU", default=False, action="store_true")
    parser.add_argument("-a", "--arch", type=str, default="vgg16", help="Specify the pretrained model to use. Supporting models: vgg16, vgg19")
    parser.add_argument("-s", "--save_dir", type=str, default="./", help="directory to save the trained model")
    args = parser.parse_args()

    print("Training will be performed with the settings as followed:")
    print("GPU: " + str(args.gpu))
    print("epochs: " + str(args.epochs))
    print("Learning Rate: " + str(args.learning_rate))
    print("Pretrained model: " + args.arch)
    print("Hidden units: " + str(args.hidden_units))
    print("Saving model to: " + str(args.save_dir))

    training_batch = 64
    validation_testing_batch = 32
    input_size = 25088
    output_size = 102
    
    # Data Preparation
    dataloader_training, dataloader_validation, dataloader_testing, dataset_training = data_preparation(args.data_container, training_batch, validation_testing_batch)

    # Download predefined model
    model = download_pretrained_model(args.arch)

    # Define Classifier
    classifier = define_classifier(input_size, args.hidden_units, output_size)

    # classifier needs to be assigned to the model before moving it to GPU
    model.classifier = classifier

    # Move model to GPU
    if args.gpu:
        model.to('cuda')

    # Optimizer should be defined after moving the model to GPU
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()

    # Train
    training(args.epochs, training_batch, validation_testing_batch, args.gpu)

    # Test
    testing(model, dataloader_testing, criterion, validation_testing_batch, args.gpu)

    # Save model
    save_checkpoint(model, args.arch, criterion, optimizer, dataset_training, args.epochs, args.save_dir)