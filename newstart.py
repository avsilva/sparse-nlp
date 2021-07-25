import sys, os, getopt
import argparse
from sparsenlp.extractdata import ExtractData 
from sparsenlp.newdatacleaner import DataCleaner

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True, help="which mode")
    ap.add_argument("-d", "--directory", required=False, help="which directory")
    ap.add_argument("-l", "--lang", required=False, help="which language")
    ap.add_argument("-t", "--table", required=False, help="which table")
    ap.add_argument("-s", "--worksheet", required=False, help="which worksheet")
    ap.add_argument("-e", "--excel", required=False, help="which excel")
    ap.add_argument("-f", "--file", required=False, help="xml file")
    ap.add_argument("-a", "--format", required=False, help="which format")
    ap.add_argument("-i", "--init", required=False, help="init date")
    ap.add_argument("-n", "--end", required=False, help="end date")
    args = vars(ap.parse_args())

    #FOLDER = 'C:/AVS/WORK/PROJECTS/sparse-nlp/data/wikipedia/'

    if args['mode'] == 'extract':
        
        print ('extracting {}'.format(args['file']))
        folder = args['directory']
        extractor = ExtractData('https://dumps.wikimedia.org/enwiki/20200820/', folder)
        xmlfile = args['file']
        extractor.download(xmlfile)
        extractor.uncompress(xmlfile)
        extractor.wikiextract(xmlfile, 'json', 4)

    elif args['mode'] == 'dataclean wiki':
        folder = args['directory']
        format = args['format']
        datacleaner = DataCleaner(folder)
        datacleaner.\
            ingest_files_into('pandasdataframe', format) .\
            explode_dataframe_into_snippets('text', '\n\n+') .\
            clean_text() .\
            all_lower() .\
            remove_stopwords(args['lang']) .\
            tokenize_text()
            #serialize('C:\AVS\WORK\PROJECTS\sparse-nlp\serialize', 'tokens.pkl')
            
            #remove_stopwords() 
        print (datacleaner.data.head())
        #print (datacleaner.data['tokens'].iloc[1]) 
        #print (len(datacleaner.data['tokens'].iloc[1])) 
        #print (datacleaner.data.dtypes)
    elif args['mode'] == 'dataclean whatsup':
        folder = args['directory']
        format = args['format']
        datacleaner = DataCleaner(folder)
        datacleaner.\
            ingest_files_into('pandasdataframe', format) .\
            autofill_dataframe(axis=0) .\
            clean_text() .\
            all_lower() .\
            remove_words_by_length(2) .\
            remove_stopwords(args['lang'], ['media', 'omitted', 'bored', 'this', 'message', 'was', 'deleted', 'i\'m']) .\
            remove_emoji() .\
            tokenize_text() .\
            save('whatsup.pkl') .\
            open('whatsup.pkl') 
        print (datacleaner.data[85:100])

        #TODO: remove stop words
        #https://www.knowledgehut.com/tutorials/machine-learning/remove-stop-words-nltk-machine-learning-python

        #TODO: improve performance with cython
        #https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html
        

if __name__ == '__main__':
    main(sys.argv[1:])

#python newstart.py -m extract -f enwiki-20200820-pages-meta-current1.xml-p1p30303.bz2 -d C:/AVS/WORK/PROJECTS/sparse-nlp/data/wikipedia/ 
#python newstart.py --mode "dataclean wiki"  --directory C:/AVS/WORK/PROJECTS/sparse-nlp/data/wikipedia/ --lang english --format json



#whatsup
#python newstart.py --mode "dataclean whatsup" --directory C:/AVS/WORK/PROJECTS/sparse-nlp/data/whatsup/ --lang portuguese --format rawtext



#https://towardsdatascience.com/benchmarking-python-nlp-tokenizers-3ac4735100c5
