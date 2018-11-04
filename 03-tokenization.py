# 2018 (c) Nicolas MAUGE, https://github.com/NicolasMauge
import spacy
import time
from utils.utils import hms_string, get_num_lines
from tqdm import tqdm

# This process takes about 1 hour on a Core i7
# nota: fastai use multiprocessing Pool to speed up the process

temp_file_dir = "data/temp/"
source_file = "temp_02_clean_text.txt"
dest_file = "temp_03_tokenized_text.csv"


def load_data(filename):
	with open(filename, "r") as file:
		data = []
		for index, article in enumerate(file):
			for line in article.rstrip().split("x_return"):
				yield line + " x_return"
			yield "\n"

def get_num_lines_s(filename):
	count = 0
	with open(filename, 'rb') as file: 
		while 1:
			buffer = file.read(8192*1024)
			if not buffer: break
			count += buffer.count(b'\n')
			count += buffer.count(b'x_return')
	return count

def tokenize_text():
	nlp = spacy.load('fr', disable=['parser', 'tagger', 'ner'])

	filename=temp_file_dir + source_file
	print(f"Tokenizing {filename}:")

	data = load_data(filename)
	
	start_time = time.time()
	with open(temp_file_dir+dest_file, "w") as file:
		for result in tqdm(nlp.pipe(data), unit=" sentence", total=get_num_lines_s(filename), ascii=True):
			tokenized = ';'.join([tok.text for tok in result])
			file.write(tokenized)

	elapsed_time = time.time() - start_time
	print("- Elapsed time: {}".format(hms_string(elapsed_time)))	

if __name__ == '__main__':
	tokenize_text()
	