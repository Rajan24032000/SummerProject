#!/usr/bin/env python
# coding: utf-8

# In[167]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')


# In[168]:


random.randint(1,train.shape[0])
train_data = np.array(train, dtype = 'float32')
test_data = np.array(test, dtype='float32')

#Define class labels for easy interpretation
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]


# In[169]:


len(test_data)


# In[170]:


i = random.randint(1,train.shape[0])
# print(i)
fig1, ax1 = plt.subplots(figsize=(2,2))
plt.imshow(train_data[i,1:].reshape((28,28)), cmap='gray') 
print("Label for the image is: ", class_names[int(train_data[i,0])])


# In[171]:


fig = plt.figure(figsize=(18,18))
ax1 = fig.add_subplot(221)
train['label'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Label')


# In[176]:


#Normalize / scale X values
X_train = train_data[:, 1:] /255.
X_test = test_data[:, 1:] /255.


# In[177]:


#Convert y to categorical if planning on using categorical cross entropy
#No need to do this if using sparse categorical cross entropy
y_train = train_data[:, 0]
y_train_cat = to_categorical(y_train, num_classes=25)

y_test = test_data[:,0]
y_test_cat = to_categorical(y_test, num_classes=25)

#Reshape for the neural network
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))


# In[129]:


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dense(25, activation = 'softmax'))


# In[130]:


model.compile(loss ='categorical_crossentropy', optimizer='adam', metrics =['acc'])
model.summary()


# In[131]:


history = model.fit(X_train, y_train_cat, batch_size = 128, epochs = 10, verbose = 1, validation_data = (X_test, y_test_cat))


# In[132]:


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[182]:


acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[178]:


# prediction = model.predict_classes(X_test)
prediction = model.predict(X_test)
prediction =np.argmax(prediction,axis=1)


# In[179]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, prediction)
print('Accuracy Score = ', accuracy)


# In[180]:


i = 7173
print(i)

plt.imshow(X_test[i,:,:,0]) 
print("Predicted Label: ", class_names[int(prediction[i])])
print("True Label: ", class_names[int(y_test[i])])


# In[139]:


type(X_test)


# In[140]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
#Print confusion matrix
cm = confusion_matrix(y_test, prediction)

fig, ax = plt.subplots(figsize=(12,12))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)


# In[141]:


#PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
fig, ax = plt.subplots(figsize=(12,12))
plt.bar(np.arange(24), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.xticks(np.arange(25), class_names)


# ## the image part
# 

# In[146]:


from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray


# In[156]:


def imageprepare(argv):
    im=Image.open(argv).convert('L')
    width=float(im.size[0])
    height=float (im.size[1])
    newImage = Image.new('L', (28,28), (255))
    
    if width >height:
        nheight = int(round((20.0 / width * height),0))
        if(nheight == 0):
            nheight=1
        img =im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop=int(round(((28-nheight)/2),0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / width * height),0))
        if(nwidth == 0):
            nwidth=1
        img=im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft=int(round(((28 - nwidth)/2),0))
        newImage.paste(img, (wleft, 4))
    tv= list(newImage.getdata())
    tva = [(255-x)* 1.0 for x in tv]
    print(tva)
    return tva

x=imageprepare('C:\\Users\\patel\\Downloads\\p_test.jpg')
# x=imageprepare('D:\\Dataset_CNN\\p_test.jpg')
print(len(x))


# In[157]:


y = np.array(x,dtype = 'int64')
training_file = y.reshape((28,28))


# In[158]:


for i in range(len(x)):
    x[i] = int(x[i])


# In[159]:


x.insert(0,21)


# In[160]:


tip = x


# In[181]:


len(tip)
print(tip)


# In[162]:


training_file


# In[163]:


# i = random.randint(1,train.shape[0])
from PIL import Image as im
fig1, ax1 = plt.subplots(figsize=(2,2))
plt.imshow(training_file[:].reshape((28,28)), cmap='gray') 
print("Label for the image is:")
plt.savefig('output.png', dpi=1000)




# In[24]:


# # prediction = model.predict_classes(X_test)
# prediction = model.predict(training_file)
# prediction =np.argmax(prediction,axis=1)


# In[46]:


# i = random.randint(1,len(prediction))
# plt.imshow(X_test[i,:,:,0]) 
# print("Predicted Label: ", class_names[int(prediction[i])])
# print("True Label: ", class_names[int(y_test[i])])


# i = random.randint(1,len(prediction))

# pred_name = CATEGORIES[np.argmax(prediction)]
plt.imshow(training_file) 
print("Predicted Label: ", class_names[np.argmax(prediction)])
print("True Label: ")


# In[27]:


import csv
  
# field names
pixel_list = []
for i in range(1,785):
    pixel = 'pixel {}'.format(i)
    pixel_list.append(pixel)


# In[28]:


rows = []
for i in range(len(x)):
    rows.append(x[i])


# In[29]:


from numpy import asarray
rss = asarray(rows)
rss_1 = rss/255.


# In[30]:


rss_11 = rss_1.reshape(28,28)


# In[31]:


filename = "external_test_data.csv"
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
      
#     # writing the fields
#     csvwriter.writerow(pixel_list)
    csvwriter.writerow(pixel_list)  
    # writing the data rows
    csvwriter.writerow(y)


# In[32]:


# prediction = model.predict(rss_11)
# prediction =np.argmax(prediction,axis=1)


# In[33]:


# i = 0
j = len(X_test)
plt.imshow(rss_11) 
print("Predicted Label: ", class_names[int(prediction[i])])
print("True Label: ", class_names[int(y_test[i])])


# In[34]:


j = len(X_test)


# In[35]:


j


# In[165]:


from csv import writer
  
# The data assigned to the list 
  
# Pre-requisite - The CSV file should be manually closed before running this code.

# First, open the old CSV file in append mode, hence mentioned as 'a'
# Then, for the CSV file, create a file object
with open('sign_mnist_test.csv', 'a', newline='') as f_object:  
    # Pass the CSV  file object to the writer() function
    writer_object = writer(f_object)
    # Result - a writer object
    # Pass the data in the list as an argument into the writerow() function
    writer_object.writerow(tip)  
    # Close the file object
    f_object.close()


# In[166]:


len(x)


# In[ ]:




