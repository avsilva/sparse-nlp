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
FOLDERPATH = './chuncks/new/'

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
    filepath = './chuncks/processed/'+newfilename
    
    file = open(filepath, 'w', encoding='utf-8') 
    for item in _data:
        file.write(str(item['id'])+','+' '.join(item['snippet'])+'\n' ) 
    file.close() 
    

def process_snippets_from_file(_parallel, _numworkers):
    
    if not os.path.isdir(FOLDERPATH):
        print ('FOLDER DOES NOT EXISTS')
        sys.exit(0)
        
    for dirpath, dirnames, filenames in os.walk(FOLDERPATH):
        allfiles = filenames

    allfiles = [file for file in allfiles if file.find('done') == -1 ]
    
    totalfiles = len(allfiles)
    filenum = 1
    for filename in allfiles:
        
        filepath = './chuncks/new/'+filename
        data = pd.read_csv(filepath, compression='bz2')
        #data = data[0:25]

        if not _parallel:

            results = clean(data)
            write_cleaned_results(filename, results)
        else:
            print ('parallel')
            data_chunks = data.to_dict('records')

            results = []
            i = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(_numworkers)) as executor:
                for chunk, cleaned_texts in zip(data_chunks, executor.map(clean, data_chunks)):
                    results +=  cleaned_texts
                    #if i % 100 == 0:
                    if i % 2 == 0:
                        print ('processing chunk '+ str(i)+ ' of file number: '+str(filenum)+' (total is: '+str(totalfiles)+')')
                    i += 1

            write_cleaned_results(filename, results)
        filenum += 1 
        os.rename('./chuncks/new/'+filename, './chuncks/new/done_'+filename)

def main(_parallel, _numworkers):
    t1=time()
    process_snippets_from_file(_parallel, _numworkers)
    t2=time()
    print("\nTime taken for processing snippets\n----------------------------------------------\n{} s".format((t2-t1)))

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print ("wrong number of arguments")
        print ("python .\process_snippets.py <parallel> <num workers>")
        sys.exit()
    main(sys.argv[1], sys.argv[2])

#python .\process_snippets.py True 4