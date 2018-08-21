import os
import json
import ast

class ModelResults():
    
    def __init__(self, folder):
        """Initializes Model results instance.

        """

        #self.path = filepath
        self.folder = folder

    def get_results(self, exception=None):
        data = []
        for subdir, dirs, files in os.walk(self.folder):
            for file in files:
                if ( file != 'log_{}'.format(exception) ):
                    with open('{}/{}'.format(subdir, file), 'r', encoding='utf-8') as handle:
                        datafile = handle.readlines()
                        for x in datafile:
                            s = ast.literal_eval(x)
                            data.append(s)
                        #data += [json.loads(x) for x in datafile]
                    
        return data

    """
    def get_results(self):
        data = []
        with open(self.path, 'r') as handle:
            datafile=handle.readlines()
            for x in datafile:
                s = ast.literal_eval(x)
                #print(type(s))
                data.append(s)
            #data += [json.loads(x) for x in datafile]
        return data
    """ 
