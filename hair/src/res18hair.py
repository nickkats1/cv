import torch
import torch.nn as nn
from torchvision import transforms,datasets,models
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import random_split
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns


device=  "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

data_path = "data"
print(os.listdir(data_path))





LABELS = ['Curly Hair', 'Straight Hair', 'Wavy Hair']

transformed_data = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])



dataset = datasets.ImageFolder(data_path,transformed_data)


train_split = int(len(dataset) * 0.8)
test_split = len(dataset) - train_split

train_data,test_data = random_split(dataset, lengths=[train_split,test_split])


BATCH_SIZE = 32
n_classes = 3
learning_rate = 0.001
epochs = 10



train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)




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


model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False


for param in model.layer4.parameters():
    param.requires_grad = True




num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features,512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512,3)
)

model.to(device)




optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    running_loss = 0
    for i,(images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(epoch+1,epochs,i+1,len(train_dataloader),loss.item())




model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    y_test = []
    y_pred = []
    for images,labels in test_dataloader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _,preds = torch.max(outputs.data,1)
        y_pred.extend(preds.cpu().numpy())
        y_test.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        print('Testing Accuracy {} %'.format(100 * correct / total))




print(classification_report(y_test,y_pred))




def plot_confusion_matrix(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cmap = sns.heatmap(cm,fmt='d',cmap="coolwarm",annot=True,xticklabels=['Curly Hair', 'Straight Hair', 'Wavy Hair'],yticklabels=['Curly Hair', 'Straight Hair', 'Wavy Hair'])
    return cmap


plt.figure(figsize=(10,6))
plot_confusion_matrix(y_test, y_pred)
plt.show()







rand_indices = np.random.choice(len(y_pred),size=min(5,len(y_pred)),replace=False)
plt.figure(figsize=(10,5 *len(rand_indices)))
for i, index in enumerate(rand_indices):
    image = img_inv(test_data[index][0])
    plt.subplot(len(rand_indices),1,i+1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    predicted_class = dataset.classes[y_pred[index]]
    true_class = dataset.classes[y_test[index]]
    color = 'green' if predicted_class == true_class else 'red'
    plt.title(f'True: {true_class} | Predicted: {predicted_class}', color=color, fontsize=12)

plt.tight_layout()
plt.show()

