import re
import spacy
import pickle
import string
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

nlp = spacy.load('en_core_web_sm')

def construct_inv_index(path, bi=False):
	'''
	:param path: path to corpus file
	:param bi: add bigrams to inverted index or not

	:returns:
	inverted-index

	saves inverted-index and doc-2-title mapping onto disk
	'''

	file = open(path, 'r')
	# doc_start_regex = re.compile(r'<doc id=\"(?P<doc_id>.+?)\".+?title=\"(?P<doc_title>.+?)\">'))
	text = file.read()
	soup = BeautifulSoup(text, 'html.parser')
	docs = soup.find_all('doc')
	inverted_index = {}
	doc_id_2_title = {}
	for match in tqdm(docs):
		doc_text = match.get_text().strip().lower()

		# remove punctuations
		doc_text = doc_text.translate(str.maketrans('', '', string.punctuation))

		doc_id = int(match.get('id'))
		doc_title = match.get('title')

		# map doc-id to title 
		doc_id_2_title[doc_id] = doc_title

		doc_text = nlp(doc_text, disable=['tagger', 'parser', 'ner'])
		stop = len(doc_text)
		for i in range(stop):
			term = doc_text[i].text.strip()

			if term in inverted_index:
				if doc_id in inverted_index[term]:
					inverted_index[term][doc_id] += 1
				else:
					inverted_index[term][doc_id] = 1
			else:
				inverted_index[term] = {doc_id: 1}

			# add bigrams to inverted-index if bi is true
			if bi and i != stop-1:
				term1 = doc_text[i+1].text.strip()
				term1 = (term, term1)
				if term1 in inverted_index:
					if doc_id in inverted_index[term1]:
						inverted_index[term1][doc_id] += 1
					else:
						inverted_index[term1][doc_id] = 1
				else:
					inverted_index[term1] = {doc_id: 1}

	with open('inv_index.pkl', 'wb+') as f:
		pickle.dump(inverted_index, f)

	with open('doc_id_2_title.pkl', 'wb+') as f:
		pickle.dump(doc_id_2_title, f)

	return inverted_index

def compute_doc_lengths(index):
	'''
	:param index: inverted-index

	saves unigram doc lengths onto disk
	'''
	doc_lengths = {}
	i = 0
	vocab_size = len(index.keys())
	for term, dic in index.items():
		if isinstance(term, tuple):
			continue
		for doc, tf in dic.items():
			wtd = 1 + np.log10(tf)
			if doc not in doc_lengths:
				doc_lengths[doc] = []

			doc_lengths[doc].append(wtd)

	for doc_id, lis in doc_lengths.items():
		doc_lengths[doc_id] = (np.array(lis)**2).sum()

	with open('doc_lengths.pkl', 'wb+') as f:
		pickle.dump(doc_lengths, f)

def compute_bi_doc_lengths(index):
	'''
	:param index: inverted-index

	saves bigram doc lengths onto disk
	'''
	doc_lengths = {}
	vocab_size = len(index.keys())
	for term, dic in index.items():
		if not isinstance(term, tuple):
			continue
		for doc, tf in dic.items():
			wtd = 1 + np.log10(tf)
			if doc not in doc_lengths:
				doc_lengths[doc] = []

			doc_lengths[doc].append(wtd)

	for doc_id, lis in doc_lengths.items():
		doc_lengths[doc_id] = (np.array(lis)**2).sum()

	with open('doc_bi_lengths.pkl', 'wb+') as f:
		pickle.dump(doc_lengths, f)


def main():
	path = input('Enter path to corpus file: ')
	inp = input('Enter 1 for Part1 and Part2, Improvement1\nEnter 2 for Part2, Improvement2\nResponse: ')
	if(inp == '1'):
		index = construct_inv_index(path)
		compute_doc_lengths(index)
	elif(inp == '2'):
		index = construct_inv_index(path, bi=True)
		compute_doc_lengths(index)
		compute_bi_doc_lengths(index)
	else:
		print('This option is not supported.')

if __name__ == "__main__":
	main()