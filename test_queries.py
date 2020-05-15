import pickle
import numpy as np
import pandas as pd
import spacy
import string
import pkg_resources
from symspellpy import SymSpell, Verbosity

nlp = spacy.load('en_core_web_sm')

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def read_query():
	'''
	:returns:
	query read from terminal
	'''
	q = input('Enter query: ')
	return q

def load_files(bi=False):
	'''
	:param bi: load bigram doc lengths or not

	:returns:
	inverted_index, unigram doc lengths, doc-2-title, [bigram doc lengths]
	'''
	with open('inv_index.pkl', 'rb+') as f:
		index = pickle.load(f)

	with open('doc_lengths.pkl', 'rb+') as f:
		doc_lengths = pickle.load(f)

	if bi:
		with open('doc_bi_lengths.pkl', 'rb+') as f:
			doc_bi_lengths = pickle.load(f)

	with open('doc_id_2_title.pkl', 'rb+') as f:
		doc_id_2_title = pickle.load(f)

	if bi:
		return [index, doc_lengths, doc_id_2_title, doc_bi_lengths]
	else:
		return [index, doc_lengths, doc_id_2_title]

def spell_correct(vocab, raw):
	'''
	:param vocab: vocab file (aka inverted-index)
	:param raw: string to be checked and corrected for spelling

	:returns:
	corrected string
	'''

	raw = raw.strip().lower().translate(str.maketrans('', '', string.punctuation))
	raw_doc = nlp(raw)

	corrected_list = []
	for tok in raw_doc:

		word = tok.text
		# if word already in vocab or a proper noun do not change
		if word in vocab:
			corrected_list.append(word)
			continue
		else:
			suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)

			try:
				# take the best suggestion if found else keep the original
				suggestion = suggestions[0].term
			except:
				suggestion = word
			corrected_list.append(suggestion)

	return ' '.join(corrected_list)

def retrieve_documents(q, files, bi=False):
	'''
	:param q: query to be searched
	:param files: list of files read from disk
	:param bi: search bigrams of query or not

	:returns:
	sorted dictionary with (doc_id, score) pairs 
	'''

	index, doc_lengths, doc_id_2_title = files[0], files[1], files[2]

	query = q.strip().lower().translate(str.maketrans('', '', string.punctuation))

	query_words = query.split()

	if bi:
		query_words = [(query_words[i], query_words[i+1]) for i in range(len(query_words)-1)]
		doc_lengths = files[3]

	query_dict = {}

	N = len(doc_lengths.keys())

	# store query terms with their frequencies
	for w in query_words:
		if w in query_dict:
			query_dict[w] += 1
		else:
			query_dict[w] = 1

	score_dict = {}
	for t, f in query_dict.items():
		try:
			posting = index[t]
		except:
			# this query word/bigram does not occur in vocab
			continue
		# tf value
		tfq = 1 + np.log10(f)
		# tf-idf value wrt query (ltc scheme)
		wtq = tfq * (np.log10(N/len(posting)))
		for doc_id, tfd in posting.items():
			# tf-idf value wrt doc (lnc scheme)
			tfd = 1 + np.log10(tfd)
			if doc_id in score_dict:
				score_dict[doc_id] += wtq * tfd
			else:
				score_dict[doc_id] = wtq * tfd

	for doc_id, score in score_dict.items():
		# divide by doc length
		score_dict[doc_id] = score/np.sqrt(doc_lengths[doc_id])

	sorted_scores = {k: v for k, v in sorted(score_dict.items(), key=lambda x: -x[1])}

	return sorted_scores

def retrieve_spell(q, files):
	'''
	:param q: query to be searched
	:param files: list of files read from disk

	:returns:
	sorted dictionary with (doc_id, score) pairs after combining results from 
	original query and corrected query
	'''

	docs_before_correction = retrieve_documents(q, files)
	corrected_q = spell_correct(files[0], q)
	docs_after_correction = retrieve_documents(corrected_q, files)
	top_docs = doc_join(docs_before_correction, docs_after_correction)

	return top_docs

def retrieve_bi(q, files):
	'''
	:param q: query to be searched
	:param files: list of files read from disk

	:returns:
	sorted dictionary with (doc_id, score) pairs after combining results from 
	original query and bigram query
	'''
	docs_uni = retrieve_documents(q, files)
	docs_bi = retrieve_documents(q, files, bi=True)
	top_docs = doc_join(docs_uni, docs_bi, bi=True)

	return top_docs

def doc_join(doc1,doc2, bi=False):
	'''
	:param doc1: dictionary of scores with original query
	:param doc2: dictionary of scores with tranformed query
	:param bi: if true we combine by adding scores else by taking max

	:returns:
	final dictionary of scores after merging the input dictionaries
	'''

	final_scores = doc1
	for doc_id,score in doc2.items():
		if doc_id in doc1:
			if bi:
				final_scores[doc_id] = doc1[doc_id] + score
			else:
				final_scores[doc_id] = max(doc1[doc_id], score)
		else:
			final_scores[doc_id] = score
	final_scores = {k:v for k, v in sorted(final_scores.items(), key=lambda kv:(-kv[1], kv[0]))}
	return final_scores

def main():
	q = read_query()
	inp = input('Enter 1 for Part1\nEnter 2 for Part2, Improvement1\nEnter 3 for Part2, Improvement2\nResponse: ')
	k = 10
	if inp in ['1', '2']:
		files = load_files()
	else:
		files = load_files(bi=True)

	if(inp == '1'):
		top_docs = retrieve_documents(q, files)
	elif(inp == '2'):
		top_docs = retrieve_spell(q, files)
	elif(inp == '3'):
		top_docs = retrieve_bi(q, files)
	else:
		print('This option is not supported.')

	# get the top k documents
	top_k_docs = [(k, v) for k, v in top_docs.items()][:k]

	if(len(top_k_docs) == 0):
		print('No relevant documents found/query terms do not exist in vocabulary')
		return

	docs = []
	scores = []
	for i in range(min(k, len(top_k_docs))):
		docs.append(files[2][top_k_docs[i][0]])
		scores.append(top_k_docs[i][1])

	df = pd.DataFrame({'Document': docs, 'Score': scores})

	print(df)

if __name__ == '__main__':
	main()