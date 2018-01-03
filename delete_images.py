import random 
import os 
imgs = os.listdir("face_images/")
for img in imgs:
	if random.random() > 0.01:
		os.remove("face_images/"+img)