import numpy as np
import h5py
import pickle 
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image 
from scipy import misc
from os import listdir 
from kevz import DebugCounter
from kevz import email                                                                                      
from keras.layers import Input, Conv3D, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import EarlyStopping 
 
y = DebugCounter()
image_dir = "face_images/"
all_images = [image_dir+file for file in listdir(image_dir)][:32]
 
dir_size = len(all_images)
training = all_images
validation = all_images[:20000]
 
 
batch_size = 32
 
def image_generator(files): 
    count = 0
    while(True):
        batch = []
        i = 0
        while i < batch_size: 
            try:
                img = misc.imread(files[count%len(files)])
                img = img.astype("float32")/255
                if img.shape == (300, 300):
                    y.GrayScale += 1
                elif img.shape != (300, 300, 3): 
                    y.Other += 1
                else:
                    batch.append(img)
                    i+=1
            except:
                y.EXCEPTIONAL_IMAGE += 1
            count+=1
        yield np.array(batch), np.array(batch)
 
input_img = Input(shape=(300, 300, 3))  # adapt this if using `channels_first` image data format
x = Conv2D(8, (3, 3), activation='relu', strides=2, padding='same')(input_img)
x = Conv2D(8, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', strides=2, padding='same')(x)
encoded = Conv2D(16, (3, 3), strides=2, activation='relu', padding='same')(x)
     
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
 
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
 
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
 
earlyStopping = EarlyStopping(patience=2)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit_generator(image_generator(training),
                steps_per_epoch=int(len(training)/batch_size), callbacks=[earlyStopping],
                verbose = 1,
                validation_data = image_generator(validation),
                validation_steps= int(len(validation)/batch_size),
                epochs=16)
 
 
 
 
autoencoder.save('autoencoder.h5')
 
#this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
 
 
 
encoder.save("encode.h5")
 
img_enc_dic = {}  
count = 0
for file in tqdm(all_images): 
    try:
        img = misc.imread(file)
        img = img.astype("float32")/255
        encoded_img = encoder.predict(np.array([img]))
        encoded_img = encoded_img.reshape(19*19*16)
        img_enc_dic[file] = encoded_img
    except:
        count+= 1
 
 
print(count/len(all_images))
 
with open("encodings.pkl", "wb") as outfile:
    pickle.dump(img_enc_dic, outfile)
 
 
#email("kevz@mit.edu", "finished encoding images", "Proportion of succesfully encrypted images: "+str(1-count/total))
#email("pmantica@mit.edu", "finished encoding images","Proportion of succesfully encrypted images: "+str(1-count/total) )
test_image = misc.imread(all_images[0])
test_image = test_image.astype("float32")/255
 
plt.imshow(test_image)
plt.show()
res = autoencoder.predict(np.array([test_image]))[0]
plt.imshow(res)
plt.show()