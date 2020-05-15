This repository is our submission to Assignment-2 for the course Information Retrieval (CS F469) offered 2nd semester 2019-2020 at BITS Pilani, Pilani Campus.

To create inverted-index and other data structures, run ```python3 util.py```

1. Enter path to corpus file (example **wiki_02** file above)
2. For part-1 and part-2, improvement1 (spelling correction) same index is used so enter 1
3. For part-2, improvement2 (phrasal queries via bigram index) new index is to be created so enter 2
4. All the files are stored in the current directory.
5. For option 1, files stored are - **inv_index.pkl**, **doc_lengths.pkl**, **doc_id_2_title.pkl**
6. For option 2, files stored are - **inv_index.pkl**, **doc_lengths.pkl**, **doc_id_2_title.pkl**, **doc_bi_lengths.pkl**
7. Notice the name of the files are same in both cases.

To query the index, run ```python3 test_queries.py```

1. Enter the query
2. To query against original index, enter 1 (should have all files with above names in the current directory)
3. To query against original index with spelling correction (improvement1), enter 2 (again should have files)
4. To query against combined index, enter 3 (should have all files from construction code option 2)

Note:
* In the test_queries.py file, the names of the files to be loaded are specified in ```load_files()``` function.
* The structure of corpus file is:
```html
<doc>...</doc>
<doc>...</doc>
...
<doc>...</doc>
```