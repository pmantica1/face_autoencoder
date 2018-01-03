import pickle
from sys import argv

infile = argv[1]
outfile = argv[2]
with open(infile, "rb") as fi:
	encodings=pickle.load(fi)

for key in list(encodings.keys()):
	formatted_key = ''.join(c for c in key if c.isdigit())
	formatted_key = int(formatted_key)
	encodings[formatted_key]  = encodings.pop(key)

with open(outfile, "wb") as of:
	pickle.dump(encodings, of)