import os
import json
import ast

class ModelResults():
    
    def __init__(self, filepath):
        """Initializes Model results instance.

        """

        self.path = filepath

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
        
