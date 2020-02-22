import re
import spacy
import pickle
import string
import numpy as np
from bs4 import BeautifulSoup

nlp = spacy.load('en_core_web_sm')

def construct_inv_index(path):

	file = open(path, 'r')
	# doc_start_regex = re.compile(r'<doc id=\"(?P<doc_id>.+?)\".+?title=\"(?P<doc_title>.+?)\">'))
	text = file.read()
	soup = BeautifulSoup(text, 'html.parser')
	docs = soup.find_all('doc')
	inverted_index = {}
	doc_id_2_title = {}
	for match in docs:
		doc_text = match.get_text().strip().lower()
		doc_id = int(match.get('id'))
		doc_title = match.get('title')
		doc_id_2_title[doc_id] = doc_title

		for tok in nlp(doc_text, disable=['tagger', 'parser', 'ner']):
			if tok.text in set(string.punctuation):
				continue
			if tok.text in inverted_index:
				if doc_id in inverted_index[tok.text]:
					inverted_index[tok.text][doc_id] += 1
				else:
					inverted_index[tok.text][doc_id] = 1
			else:
				inverted_index[tok.text] = {doc_id: 1}

	inv_index = {k:v for k, v in sorted(inverted_index.items(), key=lambda x: x[0])}

	with open('inv_index.pkl', 'wb+') as f:
		pickle.dump(inv_index, f)

	with open('doc_id_2_title.pkl', 'wb+') as f:
		pickle.dump(doc_id_2_title, f)

	print(len(inv_index.keys()))

	return inv_index

def compute_doc_vectors(index):
	doc_vectors = {}
	vocab_size = len(index.keys())
	for i, (term, dic) in enumerate(index.items()):
		for doc, tf in dic.items():
			if doc not in doc_vectors:
				doc_vectors[doc] = np.zeros((vocab_size, 1), dtype=np.float16)

			doc_vectors[doc][i] = 1 + np.log10(tf, dtype=np.float16)

	with open('doc_vectors.pkl', 'wb+') as f:
		pickle.dump(doc_vectors, f)

	print(len(doc_vectors.keys()))

def compute_doc_lengths(index):
	doc_lengths = {}
	vocab_size = len(index.keys())
	for term, dic in index.items():
		for doc, tf in dic.items():
			wtd = 1 + np.log10(tf)
			if doc not in doc_lengths:
				doc_lengths[doc] = []
				doc_lengths[doc].append(wtd)
			else:
				doc_lengths[doc].append(wtd)

	for doc_id, lis in doc_lengths.items():
		doc_lengths[doc_id] = (np.array(lis)**2).sum()

	with open('doc_lengths.pkl', 'wb+') as f:
		pickle.dump(doc_lengths, f)

	print(len(doc_lengths.keys()))

if __name__ == "__main__":
	index = construct_inv_index('./wiki_10')
	# compute_doc_vectors(index)
	compute_doc_lengths(index)