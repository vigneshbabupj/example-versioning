import numpy as np
import sys
import os
import pandas as pd
# from pandas.core.indexes.base import Index

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, datasets
import torchvision.models as models
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import glob, time

pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)

# dimensions of our images.
img_width, img_height = 150, 150
size = 150

top_model_weights_path = "model.h5"
train_data_dir = os.path.join("data", "train")
validation_data_dir = os.path.join("data", "validation")
cats_train_path = os.path.join(path, train_data_dir, "cats")
nb_train_samples = 2 * len( 
    [
        name
        for name in os.listdir(cats_train_path)
        if os.path.isfile(os.path.join(cats_train_path, name))
    ]
)
nb_validation_samples = 800
epochs = 10
batch_size = 10

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


# FilePath List
train_list = glob.glob(os.path.join(train_data_dir,"cats",'*.jpg'))
train_list.extend(glob.glob(os.path.join(train_data_dir,"dogs","*.jpg")))

val_list = glob.glob(os.path.join(validation_data_dir,"cats",'*.jpg'))
val_list.extend(glob.glob(os.path.join(validation_data_dir,"dogs","*.jpg")))

# Data Augumentation
class ImageTransform():
    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img)

# Dataset
class MyDataset(data.Dataset):
    
    def __init__(self, file_list, transform=None, phase='train'):    
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        
        img_transformed = self.transform(img, self.phase)
        
        # Get Label
        label = img_path.split('/')[-1].split('.')[0].split('\\')[-1]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0

        return img_transformed, label

class Vgg16_removehead(nn.Module):
    def __init__(self, original_model):
        super(Vgg16_removehead, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x

def predict(model, data_loader):
    global device
    model.eval()
    pred_prob = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output.shape)
            # pred = output.data.max(1, keepdim=True)[1]
            pred_prob.extend(output)

    return torch.stack(pred_prob, dim=0).cpu().numpy()


def save_bottlebeck_features():

    global net
    global dataloader_dict

    # Dataset
    train_dataset = MyDataset(train_list, transform=ImageTransform(size, mean, std), phase='train')
    val_dataset = MyDataset(val_list, transform=ImageTransform(size, mean, std), phase='val')

    # DataLoader
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

    net = models.vgg16(pretrained=True)

    # print(net)
    model_without_head = Vgg16_removehead(net)

    for param in model_without_head.parameters():
            param.requires_grad = False

    model_without_head = model_without_head.to(device)

    bottleneck_features_train = []
    # for i in tqdm(range(nb_train_samples // batch_size), desc="Train Predict vgg16..."):
    train_pred = predict(model_without_head, dataloader_dict['train'])
    bottleneck_features_train.extend(train_pred)

    bottleneck_features_validation = []
    # for i in tqdm(range(nb_validation_samples // batch_size), desc="Val Predict vgg16..."):
    val_pred = predict(model_without_head, dataloader_dict['val'])
    bottleneck_features_validation.extend(val_pred)

    np.save(open("bottleneck_features_train.npy", "wb"), bottleneck_features_train) #np.stack(bottleneck_features_train,axis=0).reshape(1000,4,4,512) ) 

    np.save(
        open("bottleneck_features_validation.npy", "wb"), bottleneck_features_validation #np.stack(bottleneck_features_validation,axis=0).reshape(800,4,4,512)
    )



def class_wise_accuracy(model,test_loader):
    classes=['cats','dogs']
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) 
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    return {'cat_acc':class_correct[0] / class_total[0],'dog_acc':class_correct[1] / class_total[1]}



def train_model(net, dataloader_dict, criterion, optimizer, num_epoch):
    
    since = time.time()
    best_acc = 0.0
    net = net.to(device)

    metrics = {"epoch":[], "accuracy":[], "loss":[], "val_accuracy":[], "val_loss":[],"cat_acc":[],"dog_acc":[]}
    
    
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-'*20)
        metrics['epoch'].append(epoch)

        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                net.train()
            else:
                net.eval()
                
            epoch_loss = 0.0
            epoch_corrects = 0
            
            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
                    
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                metrics['accuracy'].append(epoch_acc.item())
                metrics['loss'].append(epoch_loss)
            else:
                metrics['val_accuracy'].append(epoch_acc.item())
                metrics['val_loss'].append(epoch_loss)
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc                

        class_wise_acc = class_wise_accuracy(net,dataloader_dict['val'])

        metrics['cat_acc'].append(class_wise_acc['cat_acc'])
        metrics['dog_acc'].append(class_wise_acc['dog_acc'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    metrics = pd.DataFrame.from_dict(metrics)
    metrics.to_csv('Metrics.csv',index=False)

    return net


def train_top_model():

    global net
    global dataloader_dict

    # net = models.vgg16(pretrained=True)
    # net = net.to(device)

    # Specify The Layers for updating
    params_to_update = []

    update_params_name = ['classifier.6.weight', 'classifier.6.bias']

    for name, param in net.named_parameters():
        if name in update_params_name:
            param.requires_grad = True
            params_to_update.append(param)
            # print(name)
        else:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

    model = train_model(net, dataloader_dict, criterion, optimizer, epochs)

    torch.save(model, top_model_weights_path)


if __name__ == "__main__":

    global use_cuda
    global device

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    seed = 121

    torch.manual_seed(seed)

    print("Using Cuda : ", use_cuda)

    save_bottlebeck_features()
    train_top_model()


