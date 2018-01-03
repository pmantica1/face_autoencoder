from keras.models import load_model
from scipy import misc
from os import listdir
import numpy as np
import matplotlib.pyplot as plt 
import h5py


#f = h5py.File('autoencoder.h5', 'r+')
#del f['optimizer_weights']
#f.close()


image_dir = "face_images/"
all_images = [image_dir+file for file in listdir(image_dir)]
 
autoencoder = load_model("autoencoder.h5")



test_image = misc.imread(all_images[0])
test_image = test_image.astype("float32")/255

plt.imshow(test_image)
plt.show()
res = autoencoder.predict(np.array([test_image]))[0]
plt.imshow(res)
plt.show()

