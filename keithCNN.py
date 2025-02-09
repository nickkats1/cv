import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,MaxPooling2D,Dropout,BatchNormalization,Dense
from sklearn.preprocessing import LabelEncoder



seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

class_names = ['keith', 'not_keith']
class_names_labels = {class_name: i for i, class_name in enumerate(class_names)}

keith_path = "C:/cv/keithCNN/keith"
print(os.listdir(keith_path))

not_keith_path = "C:/cv/keithCNN/not_keith"
print(os.listdir(not_keith_path))

keith_richards = Image.open("C:/cv/keithCNN/keith/images (12).jpg")
plt.imshow(keith_richards)
plt.show()


not_keith = Image.open("C:/cv/keithCNN/not_keith/images (3).jpg")
plt.imshow(not_keith)
plt.show()







def display_images(folder_path, label):
    plt.figure(figsize=(15,3))
    for i, filename in enumerate(os.listdir(folder_path)[:5]):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(f'{label}')
        plt.axis('off')
    plt.show()

display_images('C:/cv/keithCNN/keith', 'keith')
display_images('C:/cv/keithCNN/not_keith', 'not_keith')




image_size=(120,120)


def load_images(folder_path, label):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: {folder_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)



keith_images, keith_labels = load_images("C:/cv/keithCNN/keith",label='keith')
not_keith_images, not_keith_labels = load_images("C:/cv/keithCNN/not_keith", label='not_keith')





X = np.concatenate([keith_images, not_keith_images], axis=0)
y = np.concatenate([keith_labels, not_keith_labels], axis=0)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10,random_state=42)

X_train = np.array(X_train,dtype="float32") / 255.0
X_test = np.array(X_test,dtype="float32") / 255.0
X_test.shape,X_train.shape


X_train = X_train.reshape((-1,120,120,3))
X_test = X_test.reshape((-1,120,120,3))

y_train = np.array(y_train)
y_test = np.array(y_test)


CNN = Sequential()
CNN.add(Conv2D(filters=32,activation='relu',kernel_size=(3,3),input_shape=(120,120,3)))
CNN.add(BatchNormalization())
CNN.add(MaxPooling2D(pool_size=(2,2)))
CNN.add(Dropout(0.3))


CNN.add(Conv2D(filters=64,activation='relu',kernel_size=(3,3)))
CNN.add(BatchNormalization())
CNN.add(MaxPooling2D(pool_size=(2,2)))
CNN.add(Dropout(0.2))

CNN.add(Conv2D(filters=128,activation='relu',kernel_size=(3,3)))
CNN.add(BatchNormalization())
CNN.add(MaxPooling2D(pool_size=(2,2)))
CNN.add(Dropout(0.1))
CNN.add(Flatten())



CNN.add(Dense(512,activation='relu'))
CNN.add(Dense(2,activation='sigmoid'))
CNN.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = CNN.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.2)
loss,acc = CNN.evaluate(X_test,y_test)
pred = CNN.predict(X_test)
loss,acc = CNN.evaluate(X_test,y_test)
print(f'testing loss: {loss:2f}')
print(f'testing accuracy: {acc*100:.2f}%')


y_pred = np.argmax(pred,axis=1)
print(classification_report(y_test, y_pred, target_names=class_names))




rand_indices = np.random.choice(len(X_test), size=min(150, len(X_test)), replace=False)
plt.figure(figsize=(15, 6 * len(rand_indices)))
for i, index in enumerate(rand_indices):
    plt.subplot(len(rand_indices), 1, i + 1)
    plt.imshow(X_test[index])
    plt.xticks([])
    plt.yticks([])
    predicted_class = class_names[int(y_pred[index])]
    true_class = class_names[int(y_test[index])]
    color = 'green' if predicted_class == true_class else 'red'
    plt.title(f'True: {true_class} | Predicted: {predicted_class}', color=color, fontsize=150)
    plt.tight_layout()
plt.show()

