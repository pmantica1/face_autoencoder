import pandas as pd 
import pickle 
import numpy as np
import csv
from sys import argv
from tqdm import tqdm

class DistanceCalculator():
	def __init__(self, path_to_encodings):
		with open(path_to_encodings, "rb") as infile:
			self.encodings = pickle.load(infile)
		self.clean()
		self.normalize()
		self.startup_df = pd.read_csv("roles.csv").dropna()
		self.startup_df["tagged_id"] = self.startup_df["tagged_id"].apply(int)
		self.startup_df["startup_id"] = self.startup_df["startup_id"].apply(int)
		self.grouped_df = self.startup_df.groupby("startup_id")
		self.types = ['founder', 'employee', 'past_investor', 'current_investor', 'advisor', 'incubator', 'board_member', 'mentor', 'attorney', 'customer']
		self.combinations = [("founder", "founder"), ("founder", "employee"), ("founder", "current_investor")]
		self.calc_all_class_distances()

	def clean(self): 
		for key in list(self.encodings.keys()):
			if isinstance(key, int):
				print("Already Cleaned...")
				break
			formatted_key = ''.join(c for c in key if c.isdigit())
			formatted_key = int(formatted_key)
			self.encodings[formatted_key] = self.encodings.pop(key)

	def normalize(self):
		for key in list(self.encodings.keys()):
			face_vector  = self.encodings.pop(key)
			normalized_face_vector = face_vector/sum(abs(face_vector))
			self.encodings[key] = normalized_face_vector 

	def calc_class_distance(self, member_type_1, member_type_2):
		filename = member_type_1+"_to_"+member_type_2+".csv"
		print("Processing "+filename)
		with open(filename, "w") as outfile:
			filenames = ["startup_id", "first_tagged_id", "second_tagged_id", "dist"]
			writer = csv.writer(outfile)
			writer.writerow(filenames)
			for startup_id, group in tqdm(self.grouped_df):
				first_df =group[group["role"] == member_type_1]
				second_df = group[group["role"] == member_type_2]
				for first_index, first_row  in first_df.iterrows():
					first_tagged_id = first_row["tagged_id"]
					for second_index, second_row in second_df.iterrows():
						second_tagged_id = second_row["tagged_id"]
						if second_tagged_id != first_tagged_id:
							dist = self.calc_ind_distance(first_tagged_id, second_tagged_id)
							if dist:
								writer.writerow([startup_id, first_tagged_id, second_tagged_id, dist])

	def calc_ind_distance(self, first_tagged_id, second_tagged_id):
		if first_tagged_id in self.encodings:
			if second_tagged_id in self.encodings:
				first_face_vector = self.encodings[first_tagged_id]
				second_face_vector = self.encodings[second_tagged_id]
				return np.sqrt(np.sum((first_face_vector-second_face_vector)**2))

	def calc_all_class_distances(self):
		for member_type_1, member_type_2 in self.combinations:
			self.calc_class_distance(member_type_1, member_type_2)

distance_calculator = DistanceCalculator(argv[1])




