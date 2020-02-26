import spacy
import pkg_resources
from symspellpy import SymSpell, Verbosity

nlp = spacy.load("en_core_web_sm")

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
raw = ("whereis th elove hehad dated forImuch of thepast who "
              "couqdn'tread in sixtgrade and ins pired him")

def convert(s): 
    # initialization of string to "" 
    str1 = " " 
  
    # using join function join the list s by  
    # separating words by str1 
    return(str1.join(s)) 

# input List of tokens
# output List of corrected tokens 
def spell_correct(raw):

	# max edit distance per lookup (per single word, not per whole input string)

	# convert list of token to string
	raw=convert(raw)

	suggestions = sym_spell.lookup_compound(raw, max_edit_distance=2)
	# display suggestion term, term frequency, and edit distance

	# for tok in nlp(raw, disable=['tagger', 'parser', 'ner']):
	# 	print(tok.__dict__)

	final_raw=""
	for suggestion in suggestions:
	    final_raw+=suggestion.term 

	return list(final_raw.split(" "))

spell_correct(list(raw.split(" ")))
