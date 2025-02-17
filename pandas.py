import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms,datasets
import os

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)



train_dir = "C:/cv/pandas/PandasBears/Train"
print(os.listdir(train_dir))
test_dir = "C:/cv/pandas/PandasBears/Test"
print(os.listdir(test_dir))


labels = ['Bears', 'Pandas']


transformed_data = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])



train_dataset = datasets.ImageFolder(train_dir,transform=transformed_data)
test_dataset = datasets.ImageFolder(test_dir,transform=transformed_data)


n_classes = len(train_dataset.classes)
learning_rate = 0.001
BATCH_SIZE = 64
epochs = 5
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader=  torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

images,labels = next(iter(train_dataloader))

images.shape

fig,axs = plt.subplots(4,8,figsize=(16,8))

for i,ax in enumerate(axs.flat):
    image = images[i].numpy().transpose((1,2,0))
    image = np.clip(image,0,1)
    ax.imshow(image)
    ax.set_title(train_dataset.classes[labels[i]])
    ax.axis("off")
plt.show()
    

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.con1 = nn.Conv2d(3, 32,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.con2 = nn.Conv2d(32, 64, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.con3 = nn.Conv2d(64, 128, kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*128, out_features=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            )
        
        
    def forward(self,x):
        x = self.pool1(F.relu(self.bn1(self.con1(x))))
        x = self.pool2(F.relu(self.bn2(self.con2(x))))
        x = self.pool3(F.relu(self.bn3(self.con3(x))))
        x = self.classifier(x)
        return x


model = Net()        
model.to(device)


optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss()
total_steps = len(train_dataloader)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.8)


for epoch in range(epochs):
    for i,(images,labels) in enumerate(train_dataloader):
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
               .format(epoch+1, epochs, i+1, total_steps, loss.item()))

    scheduler.step()




model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_dataloader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _,preds = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += torch.sum(preds == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

















