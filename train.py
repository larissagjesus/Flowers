import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Get the user inputs for the model
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type=str)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--save_dir', type = str, default='checkpoint.pth')
parser.add_argument('--arch', type = str, default='vgg19', help="Models supported: VGG, densenet, alexnet")
parser.add_argument('--learning_rate', type = float, default=0.002)
parser.add_argument('--hidden_units', type = int, default=800)
parser.add_argument('--epochs', type = int, default=2)

args = parser.parse_args()


# Assign the data directories in variables
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Define the transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_val_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(244),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_val_transforms)
val_dataset = datasets.ImageFolder(valid_dir, transform=test_val_transforms)

#  Use the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)


# Build and run the model
model = getattr(models, args.arch)(pretrained=True)


try:
    # for VGG (all)
    input_units = model.classifier[0].in_features
except:
    try:
        # for alexnet
        input_units = model.classifier.in_features 
    except:
        # for densenet (all)
        input_units = model.classifier[1].in_features
    

classifier = nn.Sequential(nn.Linear(input_units, args.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(args.hidden_units, 102),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(),lr=args.learning_rate)

if args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


epochs = args.epochs

for epoch in range(epochs):
    running_loss = 0
    test_loss = 0
    accuracy = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
            
    model.eval()
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)        
            test_loss += batch_loss.item()
                    
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            
    print("Epoch {} -> Train loss: {}, Validation loss: {}, Validation accuracy: {}".format(epoch+1,           running_loss/len(trainloader), test_loss/len(valloader), accuracy/len(valloader)))              
    
    model.train()
    
# Do validation on the test set
test_loss = 0
test_accuracy = 0

with torch.no_grad():
    for images,labels in testloader:
        images, labels = images.to(device), labels.to(device)
        log_ps = model.forward(images)
        loss = criterion(log_ps, labels)
        
        test_loss = test_loss + loss.item()
        
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy = test_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()
        
        
    print("Test loss: {}".format(test_loss/len(testloader)))
    print("Test accuracy:{}".format(test_accuracy/len(testloader)))
    
# Save the checkpoint    
checkpoint = {
    "structure": args.arch,
    "input_layers": input_units,
    "output_layers":102,
    "hidden_units": args.hidden_units,
    "model.class_to_idx": train_dataset.class_to_idx,
    "state_dic":model.state_dict(),
    "optimizer_state_dict":optimizer.state_dict(),
}

torch.save(checkpoint, args.save_dir)

