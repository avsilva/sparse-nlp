import sys
import pandas as pd
import utils.corpora as corp
from time import time
import os.path
import concurrent.futures

"""
import bz2
filepath = './chuncks/raw_437733_449053_10000.bz2'
zipfile = bz2.BZ2File(filepath) # open the file
data = zipfile.read() # get the decompressed data
newfilepath = filepath[:-4] # assuming the filepath ends with .bz2
open(newfilepath, 'wb').write(data) # write a uncompressed file
"""


def process_snippets(_limit):
    #tokenizer for snippets 
    #500 snippets -> 30 min
    engine = create_engine('postgresql://postgres@localhost:5432/sparsenlp')
    sql = "select id, text from snippets"
    sql += "where length(text) > 100 and cleaned = 'f' and random() < 0.01 limit "+str(_limit)
    data = pd.read_sql_query(sql, con=engine)
    print (data.shape)

    cleaned_texts = corp.clean_snippets(data, remove_stop_words=True, remove_punct=True, 
                   lemmas=True, remove_numbers=True, remove_spaces=True, remove_2letters_words=True, 
                remove_apostrophe=True, method='spacy', spacy_disabled_components=['tagger', 'parser'])

def clean(data):
    cleaned_texts = corp.clean_text(data, remove_stop_words=True, remove_punct=True, 
                   lemmas=True, remove_numbers=True, remove_spaces=True, remove_2letters_words=True, 
                remove_apostrophe=True, method='spacy', spacy_disabled_components=['tagger', 'parser'])
    return cleaned_texts

def write_cleaned_results(_filename, _data):
    newfilename = _filename[:-4] # assuming the filepath ends with .bz2
    filepath = './chuncks/processed/'+newfilename.replace('raw', 'processed')
    
    file = open(filepath, 'w', encoding='utf-8') 
    for item in _data:
        file.write(str(item['id'])+','+' '.join(item['snippet'])+'\n' ) 
    file.close() 
    

def process_snippets_from_file(_filename):
    #tokenizer for snippets 
    #500 snippets -> 30 min
    print (_filename)
    
    filepath = './chuncks/new/'+_filename
    if not os.path.isfile(filepath):
        print ('FILE DOES NOT EXISTS')
        sys.exit(0)

    data = pd.read_csv(filepath, compression='bz2')
    #data = data[0:25]

    parallel = True
    if not parallel:

        cleaned_texts = corp.clean_text(data, remove_stop_words=True, remove_punct=True, 
                    lemmas=True, remove_numbers=True, remove_spaces=True, remove_2letters_words=True, 
                    remove_apostrophe=True, method='spacy', spacy_disabled_components=['tagger', 'parser'])
        print (cleaned_texts)
    else:
        
        data_chunks = data.to_dict('records')

        results = []
        i = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            for chunk, cleaned_texts in zip(data_chunks, executor.map(clean, data_chunks)):
                results +=  cleaned_texts
                if i % 100 == 0:
                    print ('processing chunk '+ str(i))
                i += 1

        write_cleaned_results(_filename, results)
    os.rename('./chuncks/new/'+_filename, './chuncks/new/done_'+_filename)

def main(_filename):
    t1=time()
    process_snippets_from_file(_filename)
    t2=time()
    print("\nTime taken by processing snippets\n----------------------------------------------\n{} s".format((t2-t1)))

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print ("wrong number of arguments")
        sys.exit()
    main(sys.argv[1])

#python .\process_snippets.py raw_562836_562944_100.bz2