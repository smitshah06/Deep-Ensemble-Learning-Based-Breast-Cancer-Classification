
# Importing deepstack for powerful ensembles, [more info](https://github.com/jcborges/DeepStack)

# import splitfolders
# splitfolders.ratio('/home/bce19229/19bce229/Dataset/CDS/Class-Divided-SET/','/home/bce19229/19bce229/Dataset/CDS/Class-Divided-SET/Output',seed=1337, ratio=(.8, 0.1,0.1))

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau

img_h,img_w= (224,224)
batch_size=128
epochs=100
n_class=3

base_dir = '/home/bce19229/19bce229/Dataset/CDS/Class-Divided-SET/Output'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
         rescale=1./255,
         rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

test_datagen= ImageDataGenerator(rescale=1./255)

from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=3,
                                         cooldown=2,
                                         min_lr=1e-10,
                                         verbose=1)

callbacks = [reduce_learning_rate]
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


train_generator = train_datagen.flow_from_directory(
                    train_dir,                   # This is the source directory for training images
                    target_size=(img_h, img_w),  # All images will be resized to 300x300
                    batch_size=batch_size,
                    class_mode='categorical')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
validation_generator = test_datagen.flow_from_directory(
                        validation_dir,
                        target_size=(img_h, img_w),
                        batch_size=batch_size,
                        class_mode='categorical')

# from keras.models import load_model
# model1 = tf.keras.models.load_model('/home/bce19229/19bce229/Code/saved Models/Pretrained DenseNet121/DenseNet121-model.h5')
model2 = tf.keras.models.load_model('/home/bce19229/19bce229/Code/saved Models/Pretrained DenseNet169/DenseNet169-model.h5')
model3 = tf.keras.models.load_model('/home/bce19229/19bce229/Code/saved Models/Pretrained InceptionResNetV2/InceptionResNetV2-best-model.h5')
model4 = tf.keras.models.load_model('/home/bce19229/19bce229/Code/saved Models/Pretrained mobilenetv2/mobilenetv2-best-model.h5')
model4 = tf.keras.models.load_model('/home/bce19229/19bce229/Code/saved Models/Pretrained XceptionNet/XceptionNet-best-model.h5')

# %%
from deepstack.base import KerasMember
# with tf.device('/GPU:0'):
# member1 = KerasMember(name="model1", keras_model=model1, train_batches=train_generator, val_batches=validation_generator)
member2 = KerasMember(name="model2", keras_model=model2, train_batches=train_generator, val_batches=validation_generator)
member3 = KerasMember(name="model3", keras_model=model3, train_batches=train_generator, val_batches=validation_generator)
member4 = KerasMember(name="model4", keras_model=model4, train_batches=train_generator, val_batches=validation_generator)
# member5 = KerasMember(name="model5", keras_model=model5, train_batches=train_generator, val_batches=validation_generator)

from deepstack.ensemble import DirichletEnsemble
from sklearn.metrics import accuracy_score

wAvgEnsemble = DirichletEnsemble(N=10000, metric=accuracy_score)
wAvgEnsemble.add_members([member4, member2, member3])
wAvgEnsemble.fit()
wAvgEnsemble.describe()

from deepstack.ensemble import StackEnsemble
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

stack = StackEnsemble()

# 2nd Level Meta-Learner
estimators = [
    ('rf', RandomForestClassifier(verbose=0, n_estimators=100, max_depth=15, n_jobs=20, min_samples_split=30)),
    ('etr', ExtraTreesClassifier(verbose=0, n_estimators=100, max_depth=10, n_jobs=20, min_samples_split=20)),
    ('dtc',DecisionTreeClassifier(random_state=0, max_depth=3))
]
# 3rd Level Meta-Learner
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

stack.model = clf
stack.add_members([member4, member2, member3])
stack.fit()
stack.describe(metric=sklearn.metrics.accuracy_score)

stack.save()
stack.load()