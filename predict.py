import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np


# Get the user inputs for the prediction
parser = argparse.ArgumentParser()

parser.add_argument('image_path', type=str)
parser.add_argument('checkpoint', type=str)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--topk', type = int, default=5)
parser.add_argument('--category_names', type=str)

args = parser.parse_args()

if args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    


#Open the file with category names
import json
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
    
def load_checkpoint(filepath):
    ''' Loads the checkpoint and rebuilds the trained model. '''
    
    checkpoint = torch.load(filepath)
    
    Model = getattr(models, checkpoint['structure'])(pretrained=True)
    input_layers = checkpoint["input_layers"]
    hidden_units = checkpoint["hidden_units"]
    class_to_idx = checkpoint["model.class_to_idx"]
    state_dic = checkpoint["state_dic"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    
    
    Classifier = nn.Sequential(nn.Linear(input_layers, hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim=1))

    Model.classifier = Classifier

    Model.load_state_dict(state_dic)    
    
    return Model, class_to_idx


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
       
    transform = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(244)])
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pil_image = Image.open(image)
        
    pil_image = transform(pil_image)
    np_image = np.array(pil_image) / 255
    pil_image = (np_image - mean) / std
    pil_image = pil_image.transpose((2, 0, 1))
    
    return  torch.tensor(pil_image).to(device) #testing



def predict(image_path, checkpoint, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
       
    Model, class_to_idx = load_checkpoint(checkpoint)
    
    Model.to(device)
        
    image = process_image(image_path).unsqueeze_(0).float()
    
    Model.eval()
    with torch.no_grad():
        log_ps = Model.forward(image)
    
    ps = torch.exp(log_ps)
    top_p, top_indices = ps.topk(topk, dim=1)
        
    idx_to_class = {value:key for key,value in class_to_idx.items()}
    
    top_classes = []
   
    for i in np.array(top_indices)[0]: 
        top_classes.append(idx_to_class[i])

    return top_p, top_classes



# Running the prediction on the given image
image_path = args.image_path
checkpoint = args.checkpoint

top_p, top_classes = predict(image_path, checkpoint, args.topk)

classes_names = []

for i in top_classes:
    classes_names.append(cat_to_name[i])
    
    
image_class = cat_to_name[image_path.split("/")[2]]
top_probs_np = top_p.cpu().numpy()[0]

# Printing the image class and topk predictions
print("Image class: {}".format(image_class))
print("Top {} predictions:".format(args.topk))

for i in range(len(classes_names)):
    print("{}: {}".format(classes_names[i], top_probs_np[i]))
      
