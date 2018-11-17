# 2018 (c) Nicolas MAUGE, https://github.com/NicolasMauge
import numpy as np
from collections import Counter, defaultdict
import collections
from multiprocessing import Pool
import multiprocessing
from pathlib import Path

import time
from utils.utils import get_num_lines, hms_string, save_vocabulary, load_vocabulary
from tqdm import tqdm

max_vocab = 80000 # 60000 seems not enough for french (conjugaisons, accords, etc.)

temp_file_dir = "data/temp/"
source_file = "temp_03_tokenized_text.txt"
temp_file = "temp_04.csv"
dest_file = "data/wiki_text.csv"

"""
Hierarchy of the final file :
	- articles delimited by "\n"
		- sentences delimited by "x_return"
The file is coded with the vocabulary_stoi.pkl file
"""

def processing_data_counter(data):
	"""
	for a worker, count the words
	"""
	data_counter = Counter()
	for line in data:
		data_counter += Counter(line)

	return data_counter

def processing_data_transcode(data):
	"""
	for a worker, transcode the sentences with the vocabulary 'stoi' (sentence to index)
	"""
	data_coded = []
	up = str(stoi["_up_"])
	line_coded = []
	for token in data:
		token_coded = stoi[token.lower()]
		if token_coded!=0 and token.isupper():
			line_coded.append(up)
		line_coded.append(str(token_coded))

	return ";".join(line_coded)+"\n"



def count_words(vocab_stoi, vocab_itos):
	c = Counter()
	filename = temp_file_dir+source_file
	print(f"Word freq count in {filename}:")
	
	def load_data(filename):
		"""
		load data 1000 lines at a time
		"""
		with open(filename, "r") as file:
			data = []
			for index, line in enumerate(file):
				line = line.rstrip().lower().split(" ")
				data.append(line)
				
				if len(data) == 1000:
					yield data
					data=[]
				
			yield data

	start_time = time.time()
	iterator = load_data(filename)
	with multiprocessing.Pool() as pool:
		for result in tqdm(pool.imap_unordered(processing_data_counter, iterator), total=get_num_lines(filename)//1000, ascii=True):
			c += result

	most_common = ["_unk_", "_up_"]+[word for word,c in c.most_common(max_vocab) if c>2]
	
	elapsed_time = time.time() - start_time
	print(f"- Elapsed time: {hms_string(elapsed_time)}")

	stoi = {v:k for k,v in enumerate(most_common)}
	itos = {k:v for k,v in enumerate(most_common)}

	print("- Saving vocabularies")
	save_vocabulary(vocab_stoi, stoi)
	save_vocabulary(vocab_itos, itos)
	return stoi, itos



def transcode_text():
	filename = temp_file_dir+source_file
	print(f"Trancoding {filename}:")
	
	def load_data(filename):
		"""
		load data 1000 lines at a time
		"""
		with open(filename, "r") as file:
			for index, line in enumerate(file):
				line = line.rstrip().split(" ")
				yield line


	start_time = time.time()
	iterator = load_data(filename)
	with multiprocessing.Pool() as pool:
		with open(temp_file_dir+temp_file, "w") as file:
			for sentence in tqdm(pool.imap_unordered(processing_data_transcode, iterator), total=get_num_lines(filename), ascii=True):
				if sentence.count("0") / len(sentence) <0.1:
					file.write(sentence)

	elapsed_time = time.time() - start_time
	print(f"- Elapsed time: {hms_string(elapsed_time)}")


def vocabulary_transcode():
	vocab_stoi = "data/vocab_stoi.pkl"
	vocab_itos = "data/vocab_itos.pkl"
	global stoi, itos
	if not Path(vocab_stoi).is_file() or not Path(vocab_itos).is_file():
		stoi, itos = count_words(vocab_stoi, vocab_itos)
	else:
		stoi, itos = load_vocabulary(vocab_stoi), load_vocabulary(vocab_itos)

	stoi = defaultdict(int, stoi)

	transcode_text()


if __name__ == '__main__':
	vocabulary_transcode()
	
