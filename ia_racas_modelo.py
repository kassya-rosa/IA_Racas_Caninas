import os

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import xml.etree.ElementTree as ET

import numpy as np
np.bool = np.bool_
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split

from keras.api.models import *
from keras.api.layers import *
from keras.api.optimizers import *
from keras.api.utils import *
from keras.api.callbacks import *
from keras.api.utils import to_categorical

from keras.api.applications.densenet import DenseNet121, preprocess_input

breed_list = os.listdir("/Users/kassy/Downloads/images/Images/")

num_classes = len(breed_list)
print("{} breeds".format(num_classes))

n_total_images = 0
for breed in breed_list:
    n_total_images += len(os.listdir("/Users/kassy/Downloads/images/Images/{}".format(breed)))
print("{} images".format(n_total_images))

label_maps = {}
label_maps_rev = {}
for i, v in enumerate(breed_list):
    label_maps.update({v: i})
    label_maps_rev.update({i : v})

def show_dir_images(breed, n_to_show):
    plt.figure(figsize=(16,16))
    img_dir = "/Users/kassy/Downloads/images/Images/{}/".format(breed)
    images = os.listdir(img_dir)[:n_to_show]
    for i in range(n_to_show):
        img = mpimg.imread(img_dir + images[i])
        plt.subplot(int(n_to_show/4+1), 4, i+1)
        plt.imshow(img)
        plt.axis('off')

#%%time

# copy from https://www.kaggle.com/gabrielloye/dogs-inception-pytorch-implementation
# reduce the background noise

#os.mkdir('data')
#for breed in breed_list:
#    os.mkdir('data/' + breed)
#print('Created {} folders to store cropped images of the different breeds.'.format(len(os.listdir('data'))))

#for breed in os.listdir('data'):
#    for file in os.listdir("/Users/kassy/Downloads/annotation/Annotation/{}".format(breed)):
#        img = Image.open("/Users/kassy/Downloads/images/Images/{}/{}.jpg".format(breed, file))
#        tree = ET.parse("/Users/kassy/Downloads/annotation/Annotation/{}/{}".format(breed, file))
#        xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
#        xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
#        ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
#        ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
#        img = img.crop((xmin, ymin, xmax, ymax))
#        img = img.convert('RGB')
#        img = img.resize((224, 224))
#        img.save('data/' + breed + '/' + file + '.jpg')

def paths_and_labels():
    paths = list()
    labels = list()
    targets = list()
    for breed in breed_list:
        base_name = "data/{}/".format(breed)
        for img_name in os.listdir(base_name):
            paths.append(base_name + img_name)
            labels.append(breed)
            targets.append(label_maps[breed])
    return paths, labels, targets

paths, labels, targets = paths_and_labels()

assert len(paths) == len(labels)
assert len(paths) == len(targets)

targets = to_categorical(targets, num_classes=num_classes)

batch_size = 64

class ImageGenerator(Sequence):
    
    def __init__(self, paths, targets, batch_size, shape, augment=False):
        self.paths = paths
        self.targets = targets
        self.batch_size = batch_size
        self.shape = shape
        self.augment = augment
        
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_paths = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = np.zeros((len(batch_paths), self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
        y = np.zeros((self.batch_size, num_classes, 1))
        for i, path in enumerate(batch_paths):
            x[i] = self.__load_image(path)
        y = self.targets[idx * self.batch_size : (idx + 1) * self.batch_size]
        return x, y
    
    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        image = imread(path)
        image = preprocess_input(image)
        if self.augment:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.CropAndPad(percent=(-0.25, 0.25)),
                    iaa.Crop(percent=(0, 0.1)),
                    iaa.Sometimes(0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])
            ], random_order=True)
            image = seq.augment_image(image)
        return image
    
train_paths, val_paths, train_targets, val_targets = train_test_split(paths, 
                                                  targets,
                                                  test_size=0.15, 
                                                  random_state=1029)

train_gen = ImageGenerator(train_paths, train_targets, batch_size=32, shape=(224,224,3), augment=True)
val_gen = ImageGenerator(val_paths, val_targets, batch_size=32, shape=(224,224,3), augment=False)

inp = Input((224, 224, 3))
backbone = DenseNet121(input_tensor=inp,
                       weights="/Users/kassy/Downloads/DenseNet-BC-121-32-no-top.h5",
                       include_top=False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
outp = Dense(num_classes, activation="softmax")(x)

model = Model(inp, outp)

for layer in model.layers[:-6]:
    layer.trainable = False

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["acc"])

history = model.fit(x=train_gen, 
                    steps_per_epoch=50, 
                    validation_data=val_gen, 
                    validation_steps=50,
                    verbose=1,
                    epochs=10)

for layer in model.layers[:]:
    layer.trainable = True

# a check point callback to save our best weights
checkpoint = ModelCheckpoint('dog_breed_classifier_model.weights.h5', 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max', 
                             save_weights_only=True)

# a reducing lr callback to reduce lr when val_loss doesn't increase
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=1, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=2, min_lr=1e-7)

# for early stop
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)

history = model.fit(x=train_gen, 
                              steps_per_epoch=32, 
                              validation_data=val_gen, 
                              validation_steps=32,
                              verbose=1,
                              epochs=10,
                              callbacks=[checkpoint, reduce_lr, early_stop])
model.save('modelo_raca_caninas.h5')