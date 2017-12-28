import csv
import pandas as pd 
import numpy as np
from collections import defaultdict

types=['toxic','severe_toxic','obscene','threat','insult','identity_hate']

def read(filepath):
	content=pd.read_csv(filepath)
	texts=content['comment_text'].values

	labels=content[types].values

	for line in labels:
		if np.sum(line)>1:
			print line

	print labels.shape

	

read('train.csv')