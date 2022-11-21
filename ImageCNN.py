import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image

'''
Create our model
'''
def create_model(img_size):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32,
        kernel_size=(3, 3), activation='relu', input_shape=(img_size,img_size,3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=100, activation='relu'))   
    model.add(tf.keras.layers.Dense(units=4, activation='softmax'))    
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                    metrics=['accuracy'])

    return model

'''
Prepare data.
'''
def prep_data(path, classes, img_size):
    for i in range(len(classes)):
        
        data = read_img_data(path, classes[i], img_size)

        try:
            x = np.concatenate((x, data))
        except:
            x = data          

        # construct the onehot-encodings for a digit's data
        y_onehots = encode_onehot(i, data.shape[0])
        try:
            y = np.concatenate((y, y_onehots))
        except:
            y = y_onehots           

    return x, y

'''
Read image data
'''
def read_img_data(path, img_class, img_size):
    for file in os.listdir(path):
        if file[0] == '.':  # skip hidden files
            continue
        file_class = file.lower().split('_')[0] # skip files that are not the target class
        if file_class != img_class:
            continue

        # reading image file into memory
        img = Image.open("{}/{}".format(path, file))
        img = img.resize((img_size, img_size))

        try:
            x_train = np.concatenate((x_train, img))
        except:
            x_train = img   

    # image of various sizes, try 200x200 first.
    # images are rgb, hence 3 channels
    # -1 to let numpy computes the number of rows 
    return np.reshape(x_train, (-1, img_size, img_size, 3))     
   
'''
Performs onehot-encodings for every class (Apple, Orange, Banana, Mix).
'''
def encode_onehot(pos, n_rows):
    # 10 classes (digit 0 to 3)
    y_onehot = [0] * 4
    # create onehot-encodings for digit (i - 1)
    y_onehot[pos] = 1
    y_onehots = [y_onehot] * n_rows
    # convert python list to numpy array
    # as keras requires numpy array
    return np.array(y_onehots)

'''
Train our model.
'''
def train_model(model, x_train, y_train):
    model.fit(x=x_train, y=y_train, epochs=10)

'''
Save our model.
'''
def save_model(model, path):
    model.save(path)

'''
Load our model.
'''
def load_model(path):
    return tf.keras.models.load_model(path) 

'''
Test our model.
'''
def test_model(model, x_test, y_test):
    return model.evaluate(x=x_test, y=y_test)  


def main():
    # define constant variables
    TRAIN_DIR = './train/'
    TEST_DIR = './test/'
    CLASSES = ['apple', 'orange', 'banana', 'mixed']
    IMG_SIZE = 100

    # create our CNN model
    model = create_model(IMG_SIZE)

    # fetch training data and onehot-encoded labels
    # images are resized to IMG_SIZE x IMG_SIZE
    x_train, y_train = prep_data(TRAIN_DIR, CLASSES, IMG_SIZE)

    # normalize x_train to be between [0, 1]
    train_model(model, x_train/255, y_train)

    # # save our trained model
    # save_model(model, './imgclassifier_saved_model')

    # # showing how we can load our trained model
    # model = load_model('./ml_data/mnist_saved_model')

    # normalize y_train to be between [0, 1]
    x_test, y_test = prep_data(TEST_DIR, CLASSES, IMG_SIZE)

    # test how well our model performs against data
    # that it has not seen before
    test_model(model, x_test/255, y_test)

if __name__ == '__main__':
    main()