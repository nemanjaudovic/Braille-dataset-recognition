import random

import tensorflow as tf
import  numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras import Input, Model, Sequential, layers
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import logging
tf.get_logger().setLevel(logging.ERROR)
from keras.optimizers.legacy import Adam
import keras_tuner as kt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




# Ucitavanje slika
data = np.loadtxt("FILE.txt")
ulaz = data[:,:2352]
izlaz = data[:,2352]
ulaz = ulaz.reshape(1560,28,28,3)


# Prikaz broja odbiraka klasa
plt.figure(0)
plt.hist(izlaz,bins=26)
plt.xlabel("Redni broj klase")
plt.ylabel("Broj odbiraka u klasi")
plt.title("Prikaz broja odbiraka po klasama")
#plt.show()

# Podela na klase
lista_klasa=[]
broj_klasa=26
for i in range(broj_klasa):
    ime_klase = f'Klasa{i+1}'
    arr = ulaz[izlaz==i,:,:,:]
    globals()[ime_klase] = arr
    lista_klasa.append(arr)


# Primer svake klase
plt.figure(1)
for i in range(len(lista_klasa)):
    plt.subplot(int(len(lista_klasa)/2),2,i+1)
    plt.imshow(lista_klasa[i][0]/255.)
    plt.ylabel(f'{chr(97+i)}')
plt.show()


# Normalizacija i sredjivanje
izlaz = izlaz.reshape(1560,1)
ulaz = ulaz/255.0;


# Mesanje podataka
ind = np.random.permutation(ulaz.shape[0])
izlaz = izlaz[ind,:]
ulaz = ulaz[ind,:,:,:]
izlazOH = to_categorical(izlaz)


# Deljenje na trening, test i validacioni skup
ulaz_tr,ulaz_test,izlazOH_tr,izlazOH_test = train_test_split(ulaz,izlazOH,test_size=0.2,random_state=42)

ulaz_tr,ulaz_val,izlazOH_tr,izlazOH_val = train_test_split(ulaz_tr,izlazOH_tr,test_size=0.2,random_state=42)
print(izlazOH_tr.shape)


# Pravljenje modela

def make_model(hp):
    model = Sequential([
        layers.Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=ulaz.shape[1:4]),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,(3,3),padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(576,activation='relu'),
        layers.Dense(288, activation='relu'),
        layers.Dense(izlazOH.shape[1], activation='softmax')
    ])
    model.summary()

    lr = hp.Choice('learning_rate', values=[0.0003, 0.000325, 0.00035])
    model.compile(Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    return model



# Trazenje najboljeg hyperparametra(konstante obucavanja)
stop_early = EarlyStopping(monitor='val_accuracy',patience=5)

tuner = kt.RandomSearch(make_model,objective='val_accuracy',overwrite=True)

tuner.search(ulaz_tr,izlazOH_tr,epochs=30,batch_size=8,validation_data=(ulaz_test,izlazOH_test),callbacks=[stop_early],verbose=1)

best_model = tuner.get_best_models()
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

lr = best_hp['learning_rate']

model = tuner.hypermodel.build(best_hp)

es = EarlyStopping(monitor = 'val_loss', mode='min', verbose = 1, restore_best_weights=True)

#Treniranje modela
history = model.fit(ulaz_tr,izlazOH_tr,epochs=30,batch_size=4,validation_data=(ulaz_val,izlazOH_val),callbacks=[es],verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# Graficki prikaz greske i tacnosti na trening i validacionom skupu
plt.figure(2)
plt.plot(acc)
plt.plot(val_acc)
plt.title("Tacnost na trening i na validacionom skupu")
plt.legend(["Trening skup","Validacioni skup"])
plt.show()

plt.figure(3)
plt.plot(loss)
plt.plot(val_loss)
plt.title("Loss na trening i na validacionom skup")
plt.legend(["Trening skup","Validacioni skup"])
plt.show()


# Testiranje modela, predikcija na test skupu
print('\n')
izlazOH_pred = model.predict(ulaz_test)
izlaz_pred = np.argmax(izlazOH_pred,axis=1)
izlaz_test = np.argmax(izlazOH_test,axis=1)
A = np.sum(izlaz_pred==izlaz_test)/len(izlaz_test)
print(f'Tacnost je {A*100}%')

m=0
n=0
for i in range(len(izlaz_pred)):
    if(izlaz_pred[i]==izlaz_test[i]):
        m=i
    if(izlaz_pred[i]!=izlaz_test[i]):
        n=i
    if(m!=0 and n!=0):
        break

plt.figure(17)
plt.subplot(121)
plt.imshow(ulaz_test[m])
plt.title("Primer dobro klasifikovanog podatka")
plt.xlabel(f'Predikcija: {chr(97+izlaz_pred[m])}, tacno slovo: {chr(97 + izlaz_test[m])}')
plt.subplot(122)
plt.imshow(ulaz_test[n])
plt.title("Primer lose klasifikovanog podatka")
plt.xlabel(f'Predikcija: {chr(97+izlaz_pred[n])}, tacno slovo: {chr(97 + izlaz_test[n])}')




# Matrica konfuzije
labels = np.array([])
predictions = np.array([])
# for i in range(len(ulaz_test)):
#     predictions = np.append(predictions,np.argmax(model.predict(ulaz_test[i],verbose=0),axis=1))
# labels = np.argmax(izlazOH_test,axis=1)
# print(labels.shape)
# print(predictions.shape)
dl=[]
for i in range(26):
    dl.append(chr(i+97))

cm = confusion_matrix(y_true=izlaz_test,y_pred=izlaz_pred,normalize='true')
cmDisp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=dl)
cmDisp.plot()
plt.show()



