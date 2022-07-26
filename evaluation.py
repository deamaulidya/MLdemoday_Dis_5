# Visualize Loss & Accuracy
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import cv2
import numpy as np
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
 
 
#Fungsi untuk mengecek gambar tunggal
#Fungsi ini memberikan hasil berupa angka 0-25 yang bersesuaian dengan abjad A-Z (A=0...Z=25)
def test_image(path, model=model):
    img = image.load_img(path, target_size=(100,100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    print (classes)
    
#Fungsi untuk mengecek gambar test set (folder dengan labeled images)
#Fungsi ini memberikan hasil berupa confusion matrix dan classification report
def test_folder(path,model=model):
    #Create ImageDataGenerator to feed the model with test data folder
    test_datagen = ImageDataGenerator(
    rescale=1./255)

    #preprocess data
    test_generator = test_datagen.flow_from_directory(
        path,
        target_size=(100,100),
        class_mode='categorical',
        shuffle=False
    
    y_pred=model.predict(test_generator)
    y_pred=np.argmax(y_pred,axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes,y_pred))
    print('Classification Report')
    print(classification_report(test_generator.classes,y_pred))
    
