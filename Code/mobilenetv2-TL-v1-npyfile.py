# import all libraries
import os
import numpy as np
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
# from tf.keras.applications.xception import Xception
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Code for removing unnecessary warning logs
import warnings
warnings.filterwarnings('ignore')

#Wandb
# wandb.login()

# Wandb Weights and Bias logging init command
wandb.init(project="minor-project-mobilenet-v2", entity="minor-project")

# Loading the data from the npy file
dataDir = '/home/bce19229/19bce229/Dataset/'
train_x = np.load(dataDir+'/train_images.npy')
train_y = np.load(dataDir+'/train_labels.npy')
test_x = np.load(dataDir+'/test_images.npy')
test_y = np.load(dataDir+'/test_labels.npy')

# Printing the shape of the data to check if it is correct
print('Training Images: {} | Test Images: {}'.format(train_x.shape, test_x.shape))
print('Training Labels: {} | Test Labels: {}'.format(train_y.shape, test_y.shape))

# Data Normalization and Reshaping
print('Train: {} , {} | Test: {} , {}'.format(train_x.min(), train_x.max(), test_x.min(), test_x.max()))
train_x/=255.0
test_x/=255.0
print('Train: {} , {} | Test: {} , {}'.format(train_x.min(), train_x.max(), test_x.min(), test_x.max()))

# Class Mapping 
print('0:Benign | 1:Malignant | 2:Normal')
# Distribution of images in each class for Training-set
print(Counter(train_y))
# Distribution of images in each class for Test-set
print(Counter(test_y))

#Make Labels Categorical
train_y_oneHot = tf.one_hot(train_y, depth=3) 
test_y_oneHot = tf.one_hot(test_y, depth=3)
print('Training Labels: {} | Test Labels: {}'.format(train_y_oneHot.shape, test_y_oneHot.shape))

# initialize the training data augmentation object
trainAug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, fill_mode="nearest")

# Function with the model architecture of Transfer Learning with mobilenetv2121_Model architecture
def mobilenetv2_Model():
    baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights="imagenet", include_top=False, input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))
    # construct the head of the model that will be placed on top of the the base model
    output = baseModel.output
    output = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(output)
    output = tf.keras.layers.Flatten(name="flatten")(output)
    output = tf.keras.layers.Dense(1024, activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(512, activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(256, activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(128, activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(64, activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(32, activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(16, activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(8, activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(4, activation="relu")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(3, activation="softmax")(output)
    # place the head FC model on top of the base model (this will become the actual model we will train)
    model = tf.keras.Model(inputs=baseModel.input, outputs=output)
    # loop over all layers in the base model and freeze them so they will not be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False
    return model

# Calling the function to get the model 
model = mobilenetv2_Model()
# compile our model
print("[INFO] compiling model...")

# initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = 0.005
EPOCHS = 200
BATCHSIZE = 96 
optimizer = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

wandb.config = {
  "learning_rate": 0.005,
  "epochs": 200,
  "batch_size": 96,
  "optimizer":'adam'
}


# compile the model using binary cross-entropy rather than categorical cross-entropy 
# -- this may seem counterintuitive for multi-class classification, but keep in mind that the goal 
# here is to treat each output label as an independent Bernoulli distribution
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])
# print(model.summary())
print("Model Compiled Successfully")

# Creating a folder to save model checkpoints and logs
modelPath = '/home/bce19229/19bce229/Code/saved Models/Pretrained mobilenetv2'
if not os.path.exists(modelPath):
    os.makedirs(modelPath)
    print('Model Directory Created')
else:
    print('Model Directory Already Exists')

# Creating a callback to save the model after every epoch
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('/home/bce19229/19bce229/Code/saved Models/Pretrained mobilenetv2/mobilenetv2-best-model.h5', monitor='val_categorical_accuracy',
                                                    verbose=1, save_best_only=True, mode='auto')

# Initialising train and test size for training and testing vectors for the model_checkpoint
STEP_TRAIN = len(train_x) // BATCHSIZE
STEP_TEST = len(test_x) // BATCHSIZE


with tf.device('/GPU:0'):
    # Training the model with the training data and validation data 
    modelHistory = model.fit(trainAug.flow(train_x, train_y_oneHot, batch_size=BATCHSIZE), epochs=EPOCHS, verbose=1, callbacks=[model_checkpoint,WandbCallback()],
                            validation_data=(test_x, test_y_oneHot), shuffle = True, steps_per_epoch=STEP_TRAIN, validation_steps=STEP_TEST)

# Saving the model to the model directory
tf.keras.models.save_model(model, '/home/bce19229/19bce229/Code/saved Models/Pretrained mobilenetv2/mobilenetv2-model.h5', overwrite=True, include_optimizer=True, save_format=None,
                        signatures=None, options=None)


with tf.compat.v1.Session() as sess:
    wandb.tensorflow.log(tf.summary.merge_all())


# Plotting the training and validation accuracy and loss

# Creating directories to save the plots
losshistoryPath = './Train-Test History/Loss'
acchistoryPath = './Train-Test History/Accuracy'
if not os.path.exists(losshistoryPath) or not os.path.exists(acchistoryPath):
    os.makedirs(losshistoryPath)
    os.makedirs(acchistoryPath)

# Plot history: Loss
plt.plot(modelHistory.history['loss'], label='train data')
plt.plot(modelHistory.history['val_loss'], label='test data')
plt.title('Pretrainedmobilenetv2 Train/Test Loss')
plt.ylabel('Loss')
plt.xlabel('Number of Epochs')
plt.legend(loc="upper left")
plt.savefig('./Train-Test History/Loss/mobilenetv2-loss-Graph.png', bbox_inches = "tight")
print("Loss graph saved to Train-Test History/Loss directory")
# plt.show()

# Plot history: Accuracy
plt.plot(modelHistory.history['categorical_accuracy'], label='train data')
plt.plot(modelHistory.history['val_categorical_accuracy'], label='test data')
plt.title('Pretrainedmobilenetv2 Train/Test Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend(loc="upper left")
plt.savefig('./Train-Test History/Accuracy/mobilenetv2-Accuracy-Graph.png', bbox_inches = "tight")
print("Accuracy graph saved to Train-Test History/Accuracy directory")
# plt.show()

# Evaluate the Best Saved Model
model = tf.keras.models.load_model('/home/bce19229/19bce229/Code/saved Models   /Pretrained mobilenetv2/mobilenetv2-model.h5')
loss, accuracy, auc = model.evaluate(x=test_x, y=test_y_oneHot, batch_size=32, verbose=1)
print('Model Accuracy: {:0.2f} | Model AUC: {:.2f} | Model Loss: {:0.4f}'.format(accuracy, auc, loss))

# Function to plot the confusion matrix for the model
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Purples')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('GroundTruths')
    plt.xlabel('Predictions \n Model Accuracy={:0.2f}% | Model Error={:0.2f}%'.format(accuracy*100, misclass*100))
    plt.savefig('./Train-Test History/ConfusionMatrix/mobilenetv2-cm.png', bbox_inches = "tight")
    print("Confusion Matrix, without normalization, Plot saved to ReadMe Images Directory")
    # plt.show()

# Checking if the directory to save the confusion matrix exists or not, 
# if not then create it and save the confusion matrix
cmhistoryPath = './Train-Test History/ConfusionMatrix'
if not os.path.exists(cmhistoryPath):
    os.makedirs(cmhistoryPath)

# Predicting the test data and generating the confusion matrix
predictions = model.predict(x=test_x, batch_size=32)
predictions = tf.keras.backend.argmax(predictions, axis=-1)

# Calling the function to plot the confusion matrix
cm = confusion_matrix(test_y, predictions)
classes = ['Benign', 'Malignant', 'Normal']
plot_confusion_matrix(cm=cm, normalize = False, target_names = classes, title= "Confusion Matrix (Pretrained mobilenetv2)")

# End of the program
print("Program Completed")