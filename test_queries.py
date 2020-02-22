import pickle
import numpy as np
import pandas as pd

def read_query():
	q = input('Enter query: ')
	return q

def load_files():
	with open('inv_index.pkl', 'rb+') as f:
		index = pickle.load(f)

	with open('doc_lengths.pkl', 'rb+') as f:
		doc_lengths = pickle.load(f)

	with open('doc_id_2_title.pkl', 'rb+') as f:
		doc_id_2_title = pickle.load(f)

	return index, doc_lengths, doc_id_2_title

def retrieve_documents(q, k):

	index, doc_lengths, doc_id_2_title = load_files()

	query_words = q.strip().lower().split()
	query_dict = {}

	N = len(doc_lengths.keys())

	for w in query_words:
		if w in query_dict:
			query_dict[w] += 1
		else:
			query_dict[w] = 1

	score_dict = {}
	for t, f in query_dict.items():
		posting = index[t]
		# tf value
		tfq = 1 + np.log10(f)
		# tf-idf value wrt query
		wtq = tfq * (np.log10(N/len(posting)))
		for doc_id, tfd in posting.items():
			# tf-idf value wrt doc
			tfd = 1 + np.log10(tfd)
			if doc_id in score_dict:
				score_dict[doc_id] += wtq * tfd
			else:
				score_dict[doc_id] = wtq * tfd

	for doc_id, score in score_dict.items():
		score_dict[doc_id] = score/np.sqrt(doc_lengths[doc_id])

	sorted_scores = [(doc_id_2_title[k], v) for k, v in sorted(score_dict.items(), key=lambda x: -x[1])]

	return sorted_scores[:k]

def main():
	q = read_query()
	k = 10
	top_k_documents = retrieve_documents(q, k)

	docs = []
	scores = []
	for i in range(k):
		docs.append(top_k_documents[i][0])
		scores.append(top_k_documents[i][1])

	df = pd.DataFrame({'Document': docs, 'Score': scores})

	print(df)

if __name__ == '__main__':
	main()