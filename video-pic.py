from google.colab import drive
drive.mount('/content/drive')
import os
import cv2
from PIL import Image
from glob import glob
import numpy as np
import keras
from keras import layers
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

convert_video_to_images('/content/drive/My Drive/Colab Notebooks/MMAI5500_assignment2','/content/drive/My Drive/Colab Notebooks/MMAI5500_assignment2/assignment2_video.avi')
X, images = load_images('/content/drive/My Drive/Colab Notebooks/MMAI5500_assignment2')
print(predict(X))

def convert_video_to_images(img_folder, filename='assignment2_video.avi'):
    """
    Converts the video file (assignment2_video.avi) to JPEG images.
    Once the video has been converted to images, then this function doesn't
    need to be run again.
    Arguments
    ---------
    filename : (string) file name (absolute or relative path) of video file.
    img_folder : (string) folder where the video frames will be
    stored as JPEG images.
    """
    # Make the img_folder if it doesn't exist.'
    try:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
    except OSError:
        print('Error')
    # Make sure that the abscense/prescence of path
    # separator doesn't throw an error.
    img_folder = f'{img_folder.rstrip(os.path.sep)}{os.path.sep}'
    # Instantiate the video object.
    video = cv2.VideoCapture(filename)
    # Check if the video is opened successfully
    if not video.isOpened():
        print("Error opening video file")
    
    i = 0
    while video.isOpened():
        ret, frame = video.read()  
        if ret:
            im_fname = f'{img_folder}frame{i:0>4}.jpg'
            print('Captured...', im_fname)
            cv2.imwrite(im_fname, frame)
            i += 1
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    if i:
        print(f'Video converted\n{i} images written to {img_folder}')

def load_images(img_dir, im_width=60, im_height=44):
    """
    Reads, resizes and normalizes the extracted image frames from a folder.
    The images are returned both as a Numpy array of flattened images (i.e. the images with the 3-d shape (im_
    Arguments
    ---------
    img_dir : (string) the directory where the images are stored.
    im_width : (int) The desired width of the image.
    The default value works well.
    im_height : (int) The desired height of the image.
    The default value works well.
    Returns
    X : (numpy.array) An array of the flattened images.
    images : (list) A list of the resized images.
    """
    images = []
    fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
    fnames.sort()
    for fname in fnames:
        im = Image.open(fname)
        # resize the image to im_width and im_height.
        im_array = np.array(im.resize((im_width, im_height)))
        # Convert uint8 to decimal and normalize to 0 - 1.
        images.append(im_array.astype(np.float32) / 255.)
        # Close the PIL image once converted and stored.
        im.close()
        # Flatten the images to a single vector
    X = np.array(images).reshape(-1, np.prod(images[0].shape))
    return X, images
    
def auto(X):
    # auto encoder frame
    encoding_dim = 128
    input_img = keras.Input(shape=(X.shape[1],))
    encoded = layers.Dense(encoding_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(32, activation='relu')(encoded)
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(X.shape[1], activation='sigmoid')(decoded)
    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, encoded)
    encoded_input = keras.Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='mae')
    return autoencoder

def score(X):
    autoencoder = auto(X)
    # cross validation
    x_train, x_test = X[:int(len(X)*0.8)],X[int(len(X)*0.8):]

    # fit in model and predict
    autoencoder.fit(x_train, x_train,
                    epochs=20,
                    batch_size=150,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    prediction = autoencoder.predict(x_test)

    # visualize the prediction
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(60,44,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(prediction[i].reshape(60,44,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    # visualize the loss
    loss = autoencoder.evaluate(prediction, x_test, verbose=0)
    train_loss = tf.keras.losses.mae(prediction, x_test)
    plt.hist(train_loss, bins=50)
    plt.show()
    print('loss',loss)

    # threshold
    threshold = 0.025
    return loss, threshold

def predict(frame): 
    """ 
    Argument 
    -------- 
    frame   : Video frame with shape == (44, 60, 3) and dtype == float. 
  
    Return 
    anomaly : A boolean indicating whether the frame is an anomaly or not. 
    ------ 
    """ 
    loss, threshold = score(frame)
    # anomaly?
    if loss < threshold:
      return False
    else:
      return True