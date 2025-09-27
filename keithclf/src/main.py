import torch
import torch.nn as nn
from torchvision import transforms,datasets,models
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from PIL import Image
from sklearn.metrics import classification_report,roc_auc_score,roc_curve
import warnings
import os
import torch.nn.functional as F

warnings.filterwarnings("ignore")

device=  ("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

keith_dir = "data/keith"
print(os.listdir(keith_dir))

not_keith_dir = "data/not_keith"
print(os.listdir(not_keith_dir))

data_dir = "data"
print(os.listdir(data_dir))

labels = ['keith','not_keith']


keith = Image.open('data/keith/image8.jpeg')
plt.imshow(keith)
plt.axis("off")
plt.show()

napoleon = Image.open("data/not_keith/napolean.jpg")
plt.imshow(napoleon)
plt.axis("off")
plt.show()


transformed_data = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])




dataset = datasets.ImageFolder(data_dir,transform=transformed_data)




train_split = int(len(dataset) * 0.8)
test_split = len(dataset) - train_split

train_data,test_data = random_split(dataset, lengths=[train_split,test_split])


BATCH_SIZE = 64
epochs = 10
learning_rate = 0.001


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
    nn.Linear(512,2)
)

model.to(device)










optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss()




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
    y_pred_prob = []
    y_test = []
    for images,labels in test_dataloader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _,preds = torch.max(outputs.data,1)
        y_pred.extend(preds.cpu().numpy())
        y_test.extend(labels.cpu().numpy())
        probs = F.softmax(outputs,dim=1)[:,1].cpu().numpy()
        y_pred_prob.extend(probs)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
test_acc = 100 * correct / total
print(F"Testing Accuracy: {test_acc:.2f}%")





clf_rpt = classification_report(y_test,y_pred)
print(f"Classification Report: {clf_rpt}")




print("ROC/AUC score from resnet18 model\n")
print(f'{roc_auc_score(y_test,y_pred_prob)*100:.2f}%')



def plot_roc_cur(y_test,y_pred_prob):
    fpr,tpr,_ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Roc Curve of Fire/Not Fire Images")

plot_roc_cur(y_test, y_pred_prob)
plt.tight_layout()
plt.show()






rand_indices = np.random.choice(len(y_pred), size=min(5,len(y_pred)), replace=False)
plt.figure(figsize=(10, 5 * len(rand_indices)))
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

