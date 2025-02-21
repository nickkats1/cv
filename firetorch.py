import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms,datasets
from torch.utils.data import random_split
from PIL import Image
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve
import os
import seaborn as sns
import matplotlib.pyplot as plt
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


data_dir = "C:/fire/fire_dataset/"
print(os.listdir(data_dir))

labels = ['fire_images', 'non_fire_images']




fire = Image.open("C:/fire/fire_dataset/fire_images/fire.96.png")
plt.imshow(fire)
plt.show()


no_fire = Image.open("C:/fire/fire_dataset/non_fire_images/non_fire.97.png")
plt.imshow(no_fire)
plt.show()



data_transformed = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


mean= np.array([0.485, 0.456, 0.406]) 
std = np.array([0.229, 0.224, 0.225])

def img_inv(image):
    image = image.numpy().transpose((1,2,0))
    image = image * std + mean
    return np.clip(image,0,1)

    
    

dataset = datasets.ImageFolder(data_dir,transform=data_transformed)





train_split = int(len(dataset) * 0.8)
test_split = len(dataset) - train_split

train_data,test_data = random_split(dataset, lengths=[train_split,test_split])


BATCH_SIZE = 32
epochs = 10
learning_rate = 0.001


train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)




images,labels = next(iter(train_dataloader))



fig,axs = plt.subplots(4,8,figsize=(16,8))

for i,ax in enumerate(axs.flat):
    image = img_inv(images[i])
    ax.imshow(image)
    ax.set_title(dataset.classes[labels[i]])
    ax.axis("off")
plt.show()



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.con1 = nn.Conv2d(3, 32, kernel_size=3,padding=1)
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
            nn.Linear(32*32*128,512),
            nn.ReLU(),
            nn.Linear(512,2)
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
criterion = nn.CrossEntropyLoss().to(device)


for epoch in range(epochs):
    running_loss = 0.0
    for i,(images,labels) in enumerate(train_dataloader):
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_dataloader):.4f}")




model.eval()
y_pred = []
y_true = []
y_prob = []
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_dataloader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _,preds = torch.max(outputs.data,1)
        probs = F.softmax(outputs,dim=1)[:,1].cpu().numpy()
        y_prob.extend(probs)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
test_acc = 100 * correct / total
print(F"Testing Accuracy: {test_acc:.2f}%")


clf_rpt = classification_report(y_true,y_pred)
print(f"Classification Report: {clf_rpt}")


def plot_confusion_matrix(y_true,y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cmap = sns.heatmap(cm,fmt='d',cmap="coolwarm",annot=True,xticklabels=['fire_images', 'non_fire_images'],yticklabels=['fire_images', 'non_fire_images'])
    return cmap


plt.figure(figsize=(10,6))
plot_confusion_matrix(y_true, y_pred)
plt.show()



roc_score = roc_auc_score(y_true, y_prob)
print(f"RocAuc Score- {roc_score*100:.2f}%")


def ROC_Curve(y_true,y_prob):
    fpr,tpr,_ = roc_curve(y_true,y_prob)
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


ROC_Curve(y_true, y_prob)


rand_indices = np.random.choice(len(y_pred), size=min(5, len(y_pred)), replace=False)
plt.figure(figsize=(10, 5 * len(rand_indices)))
for i, index in enumerate(rand_indices):
    image = img_inv(test_data[index][0])
    plt.subplot(len(rand_indices),1, i + 1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    predicted_class = dataset.classes[y_pred[index]]
    true_class = dataset.classes[y_true[index]]
    color = 'green' if predicted_class == true_class else 'red'
    plt.title(f'True: {true_class} | Predicted: {predicted_class}', color=color, fontsize=12)

plt.tight_layout()
plt.show()
