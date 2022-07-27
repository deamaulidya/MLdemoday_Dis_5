# Visualize Loss & Accuracy

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn 
import numpy as np
import pandas as pd
from model import *
from sklearn.metrics import classification_report, confusion_matrix


#Fungsi untuk memvisualisasikan akurasi dan loss dari history 
def visualize(history):
    #history object created from fitting the model to training data
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
 
 
#Fungsi untuk mengecek gambar/no label
def sample_test(path, model):
    image_names=os.listdir(path)
    plt.figure(figsize=(10, 10))
    for i in range(0,len(image_names)):
        filename= image_names[i]
        images = tf.keras.utils.load_img(
            path+filename, target_size=(300,300)
        )
        x = tf.keras.utils.img_to_array(images)
        x = np.expand_dims(x, axis=0)

        img = np.vstack([x])
        res = model.predict(img)
        pred = np.argmax(res)
        abjad = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        ax = plt.subplot(4,4, i + 1)
        plt.imshow(images)
        plt.title(abjad[pred])
        plt.axis("off")
        i=i+1
    
#Fungsi untuk mengecek gambar test set (folder dengan labeled images)
#Fungsi ini memberikan hasil berupa confusion matrix dan classification report
def conf_mat(ds,model):
    class_names=ds.class_names
    actual=[]
    pred=[]
    for imgs, label in ds:
        for i in range(0,len(imgs)):
            img = imgs[i]
            img =np.expand_dims(img,axis=0)
            res= model.predict(img)
            pred.append(class_names[np.argmax(res)])
            actual.append(class_names[label[i].numpy()])
    print('Confusion Matrix')
    
    cm=confusion_matrix(y_true=actual, y_pred=pred)
    print(cm)
    df_cm = pd.DataFrame(cm, index = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
                  columns = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    print('Classification Report')
    print(classification_report(y_true=actual, y_pred=pred))
