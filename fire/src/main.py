import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import random_split
from PIL import Image
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score
import os
import seaborn as sns
import matplotlib.pyplot as plt


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


data_dir = "data"
print(os.listdir(data_dir))


labels = ['fire_images', 'non_fire_images']


fire_img = Image.open("data/fire_images/fire.96.png")
plt.imshow(fire_img)
plt.title("Fire Image")
plt.show()

no_fire_img = Image.open("data/non_fire_images/non_fire.97.png")
plt.imshow(no_fire_img)
plt.title("No Fire Image")
plt.show()


data_transformed = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])



dataset = datasets.ImageFolder(data_dir,transform=data_transformed)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset,[train_size, test_size])


BATCH_SIZE = 32
eppchs = 5
learning_rate = 0.001


train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)



def img_inv(image):
    image = image.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    image = image * std + mean
    return np.clip(image,0,1)


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
        self.con1 = nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.con2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.con3 = nn.Conv2d(64,128,kernel_size=3,padding=1)
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

epochs = 5
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
with torch.no_grad():
    correct = 0
    total = 0
    y_pred = []
    y_test = []
    y_pred_prob = []
    for images,labels in test_dataloader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _,preds = torch.max(outputs.data,1)
        probs = F.softmax(outputs,dim=1)[:,1].cpu().numpy()
        y_pred_prob.extend(probs)
        y_pred.extend(preds.cpu().numpy())
        y_test.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
test_acc = 100 * correct / total
print(F"Testing Accuracy: {test_acc:.2f}%")


    


clf_rpt = classification_report(y_test,y_pred)
print(f"Classification Report: {clf_rpt}")


roc = roc_auc_score(y_test, y_pred_prob)
print('Roc Score')
print(f'results: {roc*100:.2f}%')



def plot_roc_cur(y_test,y_pred_prob):
    fpr,tpr,_ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Roc Curve of Fire/Not Fire Images")
    plt.legend()

plot_roc_cur(y_test, y_pred_prob)
plt.tight_layout()
plt.show()




def plot_confusion_matrix(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cmap = sns.heatmap(cm,fmt='d',cmap="coolwarm",annot=True,xticklabels=['fire_images', 'non_fire_images'],yticklabels=['fire_images', 'non_fire_images'])
    return cmap


plt.figure(figsize=(10,6))
plot_confusion_matrix(y_test, y_pred)
plt.show()








rand_indices = np.random.choice(len(y_pred), size=min(5, len(y_pred)), replace=False)
plt.figure(figsize=(10, 5 * len(rand_indices)))
for i, index in enumerate(rand_indices):
    image = img_inv(test_data[index][0])
    plt.subplot(len(rand_indices),1, i + 1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    predicted_class = dataset.classes[y_pred[index]]
    true_class = dataset.classes[y_test[index]]
    color = 'green' if predicted_class == true_class else 'red'
    plt.title(f'True: {true_class} | Predicted: {predicted_class}', color=color, fontsize=12)

plt.tight_layout()
plt.show()
