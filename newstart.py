import sys, os, getopt
import argparse
from sparsenlp.extractdata import ExtractData 
from sparsenlp.newdatacleaner import DataCleaner

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True, help="which mode")
    ap.add_argument("-t", "--table", required=False, help="which table")
    ap.add_argument("-s", "--worksheet", required=False, help="which worksheet")
    ap.add_argument("-e", "--excel", required=False, help="which excel")
    ap.add_argument("-f", "--file", required=False, help="xml file")
    ap.add_argument("-i", "--init", required=False, help="init date")
    ap.add_argument("-n", "--end", required=False, help="end date")
    args = vars(ap.parse_args())

    FOLDER = 'C:/AVS/WORK/PROJECTS/sparse-nlp/data/'

    if args['mode'] == 'extract':
        
        print ('extracting {}'.format(args['file']))
        extractor = ExtractData('https://dumps.wikimedia.org/enwiki/20200820/', FOLDER)
        xmlfile = args['file']
        extractor.download(xmlfile)
        extractor.uncompress(xmlfile)
        extractor.wikiextract(xmlfile, 'json', 4)

    elif args['mode'] == 'dataclean':
        datacleaner = DataCleaner(FOLDER)
        datacleaner.ingest_files_into('pandasdataframe', 'json')
        

if __name__ == '__main__':
    main(sys.argv[1:])

#python newstart.py -m extract -f enwiki-20200820-pages-meta-current1.xml-p1p30303.bz2
#python newstart.py -m dataclean 
