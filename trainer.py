import torch as T
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import os

#Stops it from using gradiant descent on these parameters
def setParameterRequiresGrad(model, extracting): 
    if extracting:
        for param in model.parameters():
            param.requiresGrad = False

def trainModel(model, dataLoaders, criterion, optimizer, epochs=25):
    for epoch in range(epochs):
        print('Epoch % /&d' % (epoch, epochs -1))
        print('-'*15)
        
        #When pytorch is in training mode it collects data for batch normalizatoin
        #So we need to handle it in case it is meant to be in eval mode
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0

            for inputs, labels in dataLoaders[phase]:
                inputs = inputs.to(device) #PyTorch needs to know if is using GPU or CPU
                labels = labels.to(device)

                optimizer.zero_grad() #Zero gradiant for optimizer

                #Sets gradiants to enabled to disabled, which we only want when in training mode
                with T.set_grad_enabled(phase=='train'): 
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = T.max(outputs, labels) #Prediction class for our outputs

                    if phase == 'Train': #Back propogate + step optimizer if training
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0) #Running loss is float & loss is tensor so need to deregulation
                correct += T.sum(preds == labels.data) #Predictions = Label, so if prediction is right

        epoch_loss = running_loss / len(dataLoaders[phase].dataset) #This runs the accuracy of the guesses I think
        epoch_acc = correct.double() / len(dataLoaders[phase].dataset) #The .double converts tensor to float 64

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

if __name__ == '__main__':
    root_dir = 'hymenoptera_data/'

    image_transforms = {
        'train': transforms.Compose([transforms.RandomRotation((-270, 270)),
                 transforms.Resize((224,224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.224])]),
        
        'val': transforms.Compose([transforms.RandomRotation((-270, 270)),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.224])])}

    dataGenerator = {
    k: datasets.ImageFolder(os.path.join(root_dir, k), image_transforms[k])
    for k in ['train', 'val']
    }

    dataLoader = {k: T.utils.data.DataLoader(dataGenerator[k], batch_size=2, shuffle=True, num_workers = 4) for k in ['train', 'val']}

    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') #Uses gpu if available, otherwise cpu
    model = models.resnet18(pretrained=True)

    setParameterRequiresGrad(model, True) #If extracting is true, we set the the function to False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            params_to_update.append(param)
            print('\t', name)
    
    trainModel(model, dataLoader, criterion, optimizer)