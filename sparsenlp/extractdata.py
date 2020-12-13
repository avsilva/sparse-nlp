import requests
import bz2
import os

#BASEFOLDER = "C:/AVS/WORK/PROJECTS/sparse-nlp/data/"

class ExtractData():
    """
    An extraction of text data.
    """

    def __init__(self, baseurl, basefolder):
        self.basefolder = basefolder
        self.baseurl = baseurl

    def download(self, file):
        url = self.baseurl+file
        r = requests.get(url, allow_redirects=True)
        open(self.basefolder+file, 'wb').write(r.content)

    def uncompress(self, file):
        zipfile = bz2.BZ2File(self.basefolder+file)
        data = zipfile.read() # get the decompressed data
        newfilepath = file[:-4] # assuming the filepath ends with .bz2
        open(self.basefolder+newfilepath, 'wb').write(data) # write a uncompressed file

    def wikiextract(self, file, format=None, processes=None):
        """
        Extracts data from wikidump xml file using WikiExtractor:
        https://github.com/attardi/wikiextractor

        :param file: xml file to process
        :param format: output format file
        :param processes: number of processes
        """
        
        cmd = "python -m wikiextractor.WikiExtractor "+self.basefolder+file
        if format == 'json':
            cmd += " --json"
        if processes is not None:
            cmd += " --processes "+str(processes)
        cmd += " --output "+self.basefolder
        os.system(cmd)


#https://dumps.wikimedia.org/enwiki/20200820/enwiki-20200820-pages-meta-current1.xml-p1p30303.bz2