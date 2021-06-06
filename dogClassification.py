import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import os
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import ImageFile


##---------------------------
# STEP 0: Checks
##---------------------------

print("~~~ Running dog classification ~~~")

# load filenames for human and dog images
human_files = np.array(glob("lfw/*/*"))
dog_files = np.array(glob("dogImages/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))



##---------------------------
# STEP 1: Detect Humans
##---------------------------

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
#plt.imshow(cv_rgb)
#plt.show()

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]
humanFiles_faces = 0
dogFiles_faces = 0

for f in human_files_short:
    if(face_detector(f)):
        humanFiles_faces += 1

for f in dog_files_short:
    if(face_detector(f)):
        dogFiles_faces += 1

print("{}% of humanFiles have faces".format((humanFiles_faces*len(human_files_short)/100)))        
print("{}% of dogFiles have faces".format((dogFiles_faces*len(dog_files_short)/100)))    

##---------------------------
# STEP 2: Detect Dogs
##---------------------------

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
    
# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    VGG16.eval()
    transform_params = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    img = Image.open(img_path)
    img = transform_params(img).float()
    img = img.unsqueeze(0)
    
    if use_cuda:
        img = img.cuda()
    
    predictions = VGG16(img)
    j, index = predictions.max(1)
    
    return index.item() # predicted class index


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    
    ## TODO: Complete the function.
    dog_index = VGG16_predict(img_path)

    if(dog_index >= 151 and dog_index <= 268):
        return True
    
    return False # true/false


### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
humanFiles_dogs = 0
dogFiles_dogs = 0

for f in human_files_short:
    if(dog_detector(f)):
        humanFiles_dogs += 1
for f in dog_files_short:
    if(dog_detector(f)):
        dogFiles_dogs += 1

print("{}% of humanFiles have dogs".format((humanFiles_dogs*len(human_files_short)/100)))        
print("{}% of dogFiles have dogs".format((dogFiles_dogs*len(dog_files_short)/100)))      



##---------------------------
# STEP 3: Dogs CNN
##---------------------------
### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

dogsDir = 'dogImages'

#transform_params = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
#                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_params = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

#transform_params = transforms.Compose([transforms.RandomRotation(10),
#                                transforms.RandomResizedCrop(224),
#                                transforms.RandomHorizontalFlip(),
#                                transforms.ToTensor(),
#                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])])

training_data = datasets.ImageFolder('dogImages/train/', transform=transform_params)
validation_data = datasets.ImageFolder('dogImages/test/', transform=transform_params)
testing_data = datasets.ImageFolder('dogImages/valid/', transform=transform_params)

batch_size = 20
num_workers = 0

training_loader = DataLoader(training_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
testing_loader = DataLoader(testing_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

loaders_scratch = {'train':training_loader, 'valid':validation_loader, 'test':testing_loader}

print("{} training images".format(len(training_loader)))
print("{} validation images".format(len(validation_loader)))
print("{} testing images".format(len(testing_loader)))


# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        
        '''
        ## Define layers of a CNN
        self.conv_1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.batchNorm_1 = nn.BatchNorm2d(16)
        
        self.conv_2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.batchNorm_2 = nn.BatchNorm2d(32)
        
        self.conv_3 = nn.Conv2d(32, 64, 3, padding = 1)
        self.batchNorm_3 = nn.BatchNorm2d(64)

        self.conv_4 = nn.Conv2d(64, 128, 3, padding = 1)
        self.batchNorm_4 = nn.BatchNorm2d(128)

        self.conv_5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.batchNorm_5 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.25)
        
        self.fc_1 = nn.Linear(256*7*7, 1024)
        self.batchNorm_fc_1 = nn.BatchNorm1d(1024)
        
        #Num of dog breed classes = 133
        self.fc_2 = nn.Linear(1024, 133)
        '''
        
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 500)
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 133)
        self.dropout = nn.Dropout(0.25)
        
        
    def forward(self, x):
        ## Define forward behavior
        
        '''
        # Conv layer with relu -> maxpool2d -> batch normalize
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.batchNorm_1(x)
        
        x = self.pool(F.relu(self.conv_2(x)))
        x = self.batchNorm_2(x)

        x = self.pool(F.relu(self.conv_3(x)))
        x = self.batchNorm_3(x)
        
        x = self.pool(F.relu(self.conv_4(x)))
        x = self.batchNorm_4(x)
        
        x = self.pool(F.relu(self.conv_5(x)))
        x = self.batchNorm_5(x)
        
        x = self.dropout(x.view(-1, 256*7*7))
        x = self.dropout(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        '''
        # First Convolutional Layer with ReLU activation, 
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(self.conv3_bn(F.elu(self.conv3(x))))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(x)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

#-#-# You do NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
    
    
### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.Adagrad(model_scratch.parameters(), lr=0.01)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased from {:.6f} to {:.6f}. Saving model... '.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model_scratch.pt')
            valid_loss_min = valid_loss
            
    # return trained model
    return model


# train the model
model_scratch = train(10, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))