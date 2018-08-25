# Import libraries
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image


def load_model(filepath):
    print("Loading the pretrained model...")

    loaddata = torch.load(filepath)

    #... (restore model from state)
    epochs = loaddata['epochs']
    model_name = loaddata['model_name']
    classifier = loaddata['classifier']

    model = download_pretrained_model(model_name)
    model.classifier = classifier
    
    model.class_to_idx = loaddata['class_to_idx']
    model.load_state_dict(loaddata['model_state'])
    
    return model


def label_mapping(input_file):
    print("Mapping the labels...")
    with open(input_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def download_pretrained_model(model_name):
    model = None
    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    if model_name == "vgg19":
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    return model


def process_image(image):
    print("Processing the image...")

    # scale
    img_w, img_h = image.size
    
    if(img_w > img_h):
        image = image.resize(size=(int((img_w*256)/img_h),256))
    elif(img_w < img_h):
        image = image.resize(size=(256,int((img_w*256)/img_h)))

    # crop
    img_w_new, img_h_new = image.size
    c1 = int(img_w_new/2-112)
    c2 = int(img_h_new/2-112)
    c3 = int(img_w_new/2+112)
    c4 = int(img_h_new/2+112)
    
    image = image.crop((c1, c2, c3, c4)) # Getting (224, 224) image
    
    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = np.array(image) / 255
    image_norm = (image_array - mean) / std
    
    # reorder dimension
    image_trans = image_norm.transpose((2,0,1))
    
    return torch.from_numpy(image_trans) # converting ndarray to tensor



def load_image(filepath):
    print("Loading the pretrained model...")

    img_pil = Image.open(filepath)
    output = process_image(img_pil)
    return output, img_pil



def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, topk, label, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print("Predicting.......")
    output = process_image(image_path)
    output.unsqueeze_(0)

    if gpu:
        output = output.cuda().float()
    else:
        output = output.float()

    model.eval()
    
    with torch.no_grad():
        score = model(output)
        topResults = torch.topk(score, topk)
    
    prob, classes = topResults

    cat_to_name = label_mapping(label)
    names = match(cat_to_name, classes)
    result(names, prob[0], output)



def match(cat_to_name, classes):
    # match flower names with the chosen index
    classes = np.array(classes)
    names = []
    for num in classes[0]:
        names.append(cat_to_name[str(num)])
    return names



def result(names, prob, output):

    # matplotlib not supported in Udacity's web interface
    # y_pos = names
    # x_pos = prob[0]

    # ax_img = imshow(output)
    # ax_img.set_title(names[0])

    # plt.figure(figsize=(4,4))
    # plt.barh(range(len(y_pos)), np.exp(x_pos))
    # plt.yticks(range(len(y_pos)), y_pos)

    # plt.show()

    print(names)
    print(np.exp(prob))
    print("Provided image is ... " + names[0] + " with " + str(np.exp(prob[0])) + " probability.")


if __name__ == "__main__":
    # Example: python predict.py -k 5 -m ./trained_model.pth -i ./flowers/test/10/image_07090.jpg -l ./cat_to_name.json
    
    parser = argparse.ArgumentParser(description="Find the name of flower from the given image")
    # Required
    parser.add_argument("-k", "--topk", type=int, required=True, help="number of top probability items to pick from the prediction")
    parser.add_argument("-m", "--model", type=str, required=True, help="filepath and filename of saved model")
    parser.add_argument("-i", "--image", type=str, required=True, help="filepath and filename of image to predict")
    parser.add_argument("-l", "--label", type=str, required=True, help="filepath and filename of labels to map with index")
    # Optional
    parser.add_argument("-g", "--gpu", help="use GPU", default=False, action="store_true")

    args = parser.parse_args()

    print("Prediction is in progress...")

    # load model
    model = load_model(args.model)
    if args.gpu:
        model.to('cuda')
    else:
        model.to('cpu')

    # load image
    output, img_pil = load_image(args.image)
     
    # predict
    predict(img_pil, model, args.topk, args.label, args.gpu)

    # # label mapping
    # cat_to_name = label_mapping(args.label)

    # # match index with category name
    # names = match(cat_to_name, classes)

    # # result
    # result(names, prob[0], output)