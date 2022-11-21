import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image

'''
Create our model
'''
def create_model(img_size, img_mode):
    model = tf.keras.Sequential()
    if img_mode == 'RGB':
        channels = 3
    else:
        channels = 4
    model.add(tf.keras.layers.Conv2D(filters=32,
        kernel_size=(3, 3), activation='relu', input_shape=(img_size,img_size,channels)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu')) 
    model.add(tf.keras.layers.Dense(units=4, activation='softmax'))    
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                    metrics=['accuracy'])

    return model

'''
Prepare data.
'''
def prep_data(path, classes, img_size, img_mode):
    for i in range(len(classes)):
        
        data = read_img_data(path, classes[i], img_size, img_mode)
        print(data.shape)

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
def read_img_data(path, img_class, img_size, img_mode):
    for file in os.listdir(path):
        if file[0] == '.':  # skip hidden files
            continue
        file_class = file.lower().split('_')[0] # skip files that are not the target class
        if file_class != img_class:
            continue

        # reading image file into memory
        img = Image.open('{}/{}'.format(path, file)).convert(mode=img_mode)
        img = img.resize((img_size, img_size))

        try:
            x_train = np.concatenate((x_train, img))
        except:
            x_train = img

    # images are rgb, hence 3 channels
    # if mode is rgba or cmyk, will be 4 channels
    if img_mode == 'RGB':
        channels = 3
    else:
        channels = 4
    # -1 to let numpy computes the number of rows 
    return np.reshape(x_train, (-1, img_size, img_size, channels))     
   
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
    
'''
Do our own evaluation; printing out predictions given by our model.
'''
def manual_eval(model, x_test, y_test_1hot):
    # get predicted values from model
    predictions = model.predict(x=x_test)

    # eyeball predicted values against actual ones
    for i in np.arange(len(predictions)):
        print('Actual: ', y_test_1hot[i], 'Predicted: ', predictions[i])        

    # compute accuracy
    n_preds = len(predictions)       
    correct = 0
    wrong = 0

    for i in np.arange(n_preds):
        pred_max = np.argmax(predictions[i])
        actual_max = np.argmax(y_test_1hot[i])
        if pred_max == actual_max:
            correct += 1
        else:
            wrong += 1
    
    print('correct: {0}, wrong: {1}'.format(correct, wrong))
    print('accuracy =', correct/n_preds)

def main():
    # define constant variables
    TRAIN_DIR = './train/'
    TEST_DIR = './test/'
    CLASSES = ['apple', 'orange', 'banana', 'mixed']
    IMG_SIZE = 100 # use 100px as initial image size. can be modified to experiment
    IMG_MODE = 'RGBA' # convert all images to RGB. Can experiment with others e.g. RGBA, CMYK.

    # create our CNN model
    model = create_model(IMG_SIZE, IMG_MODE)

    # fetch training data and onehot-encoded labels
    # images are resized to IMG_SIZE x IMG_SIZE
    x_train, y_train = prep_data(TRAIN_DIR, CLASSES, IMG_SIZE, IMG_MODE)

    # normalize x_train to be between [0, 1]
    train_model(model, x_train/255, y_train)

    # # save our trained model
    # save_model(model, './imgclassifier_saved_model')

    # # showing how we can load our trained model
    # model = load_model('./ml_data/mnist_saved_model')

    # normalize y_train to be between [0, 1]
    x_test, y_test = prep_data(TEST_DIR, CLASSES, IMG_SIZE, IMG_MODE)

    # test how well our model performs against data
    # that it has not seen before
    test_model(model, x_test/255, y_test)

    manual_eval(model, x_test/255, y_test)

if __name__ == '__main__':
    main()