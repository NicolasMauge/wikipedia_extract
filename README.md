# wikipedia_extract
This repository gathers some shells and python scripts to extract and clean wikipedia text for machine learning purposes

## Prerequisites:
Anaconda package (tested on >5.1 but probably works on previous versions)

## 01_wikipedia_download.sh
This script, based on the facebook repository, downloads the XML dump of wikipedia in a chosen language

## 02_extract_text.py
This script extracts the articles in a file using some basic cleanings. 
- Each article is delimited by '\n'
- In each articles, each paragraph is delimited by "x_return"

## 03_tokenization.py
This script uses SpaCy to tokenize the text.
Duration on i7: 1h. 
- I tried to use multiprocessing.Pool with no success

## 04_codage.py
This script:
- builds the vocabulary (how to code a word in integer)
- transcodes the text

