from keras.models import load_model
from scipy import misc
from os import listdir
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
import pickle 

image_dir = "face_images/"
all_images = [image_dir+file for file in listdir(image_dir)]
print(len(all_images))
encoder = load_model("encode.h5")
img_enc_dic = {}  
count = 0
for file in tqdm(all_images): 
	try:
		img = misc.imread(file)
		img = img.astype("float32")/255
		encoded_img = encoder.predict(np.array([img]))
		encoded_img = encoded_img.reshape(38*38*8)
		img_enc_dic[file] = encoded_img
	except:
		count+= 1 

print(count/len(all_images))

d1 = dict(d.items()[len(d)/2:])
d2 = dict(d.items()[:len(d)/2])

pickle.dump(d1, open("encodings_part_1.pkl", "wb"))
pickle.dump(d2, open("encodings_part_2.pkl", "wb"))



