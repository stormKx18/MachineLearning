#Source: https://www.youtube.com/watch?v=UkzhouEk6uY&index=3&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL
#PREPROCESS DATA************************************
#--------------------------------------------------
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
#--------------------------------------------------

#--------------------------------------------------
train_labels=[]
train_samples=[]
#--------------------------------------------------


#--------------------------------------------------
#Create dataset
#Add lower and upper limit samples (13 & 100)
train_labels.append(0)
train_samples.append(13)

train_labels.append(1)
train_samples.append(100)

#Add 100 datapoints
for i in range(50):
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older= randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

#Add 2000 datapoints
for i in range(1000):
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older= randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)


#--------------------------------------------------


#--------------------------------------------------
#print raw data
for i in train_samples:
    print(i)
#--------------------------------------------------


#--------------------------------------------------
#transform to arrays
train_labels=np.array(train_labels)
train_samples=np.array(train_samples)
#--------------------------------------------------


#--------------------------------------------------
#normalize dataset
scaler= MinMaxScaler(feature_range=(0,1))
scaled_train_samples= scaler.fit_transform((train_samples).reshape(-1,1))
#--------------------------------------------------


#--------------------------------------------------
#Print normalized data
for i in scaled_train_samples:
    print(i)
#--------------------------------------------------

#NN model in keras************************************

#--------------------------------------------------
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
#--------------------------------------------------


#--------------------------------------------------
#Create NN model
model = Sequential([
    Dense(16, input_shape=(1,),activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
#--------------------------------------------------


#--------------------------------------------------
model.summary()
#--------------------------------------------------


#--------------------------------------------------
#compile model
model.compile(Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#--------------------------------------------------


#--------------------------------------------------
#Train model
model.fit(scaled_train_samples,train_labels, validation_split=0.1,batch_size=10, epochs=100, shuffle=True, verbose=2)
#--------------------------------------------------

#Make predictions ************************************

#--------------------------------------------------
#Preprocess test data
test_labels=[]
test_samples=[]
#--------------------------------------------------


#--------------------------------------------------
#Create dataset
#Add lower and upper limit samples (13 & 64)
test_labels.append(0)
test_samples.append(13)

test_labels.append(1)
test_samples.append(100)

#Add 100 datapoints
for i in range(50):
    random_younger=randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    random_older= randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

#Add 2000 datapoints
for i in range(1000):
    random_younger=randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    random_older= randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)


#--------------------------------------------------

#--------------------------------------------------
#transform to arrays
test_labels=np.array(test_labels)
test_samples=np.array(test_samples)
#--------------------------------------------------


#--------------------------------------------------
#normalize dataset
scaler= MinMaxScaler(feature_range=(0,1))
#MinMaxScaler object that was used on the training samples should also be used on the test samples
scaled_test_samples= scaler.fit_transform((train_samples).reshape(-1,1))
#--------------------------------------------------


#--------------------------------------------------
#Predict
predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)
for i in predictions:
    print(i)

rounded_predictions=model.predict_classes(scaled_test_samples,batch_size=10,verbose=0)
for i in rounded_predictions:
    print(i)
#--------------------------------------------------

#CONFUSION MATRIX ****************************************

#--------------------------------------------------
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
#--------------------------------------------------

#--------------------------------------------------
#Confusion matrix
cm = confusion_matrix(test_labels, rounded_predictions)
#--------------------------------------------------

#--------------------------------------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
#--------------------------------------------------

#--------------------------------------------------
cm_plot_labels=["no_side_effects","had_side_effects"]
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
#--------------------------------------------------


#--------------------------------------------------
#Save Keras Model
model.save('/home/chrisxt/Documents/MachineLearning/keras/models/medical_trial_model.h5')
#Saves: architecture, weights, loss, optimizer, state of optimizer
#allowing to resume training
#--------------------------------------------------

#--------------------------------------------------
#Load model
from keras.models import load_model
new_model=load_model('/home/chrisxt/Documents/MachineLearning/keras/models/medical_trial_model.h5')
#--------------------------------------------------

#--------------------------------------------------
new_model.summary()
new_model.get_weights()
new_model.optimizer
#--------------------------------------------------

#--------------------------------------------------
#Model to json
#Only saves the architecture of a model not its weights or
#training configuration
json_string=model.to_json()

#Save as YAML
#yaml_string=model.to_yaml()
#--------------------------------------------------


#--------------------------------------------------
#Model reconstruction from json
from keras.models import model_from_json
model_architecture= model_from_json(json_string)

#model reconstruction from yaml
#from keras.models import model_from_yaml
#model=model_from_yaml(yaml_string)

model_architecture.summary()
#--------------------------------------------------

#--------------------------------------------------
#Save only the weights of the model
model.save_weights('/home/chrisxt/Documents/MachineLearning/keras/models/my_model_weights.h5')
#--------------------------------------------------

#--------------------------------------------------
#Load weights into model
model2= Sequential([
    Dense(16, input_shape=(1,),activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model2.load_weights('/home/chrisxt/Documents/MachineLearning/keras/models/my_model_weights.h5')
#--------------------------------------------------

#--------------------------------------------------

#--------------------------------------------------


#--------------------------------------------------

#--------------------------------------------------

#--------------------------------------------------

#--------------------------------------------------


#--------------------------------------------------

#--------------------------------------------------

#--------------------------------------------------

#--------------------------------------------------


#--------------------------------------------------

#--------------------------------------------------

#--------------------------------------------------

#--------------------------------------------------


#--------------------------------------------------

#--------------------------------------------------
