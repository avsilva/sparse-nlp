import sys
from scipy.spatial import distance


def cosine():

    for arr in compare:
        dist = distance.cosine(A, arr)
        print ('cosine distance between {} and {} is: {}'.format(A, arr, round(dist, 2)))

    
def euclidean():
    for arr in compare:
        dist = distance.euclidean(A, arr)
        print ('euclidean distance between {} and {} is: {}'.format(A, arr, round(dist, 2)))


def cityblock():
    for arr in compare:
        dist = distance.cityblock(A, arr)
        print ('cityblock distance between {} and {} is: {}'.format(A, arr, round(dist, 2)))

def jaccard():
    for arr in compare:
        dist = distance.jaccard(A, arr)
        print ('jaccard distance between {} and {} is: {}'.format(A, arr, round(dist, 2)))

if __name__ == '__main__':

    METRICS = {'cosine': cosine, 'euclidean': euclidean, 'cityblock': cityblock,
    'jaccard': jaccard}

    A = [1, 0, 0, 0]
    B = [0, 1, 0, 0]
    C = [0, 0, 1, 0]
    D = [0, 0, 0, 1]
    E = [1, 1, 0, 0]
    F = [1, 0, 0, 0]
    compare = [B, C, D, E, F]

    if len(sys.argv) != 2:
        print ("wrong number of arguments")
        print ("python .\metrics.py <mode>")
        sys.exit()

    metric = sys.argv[1]
    METRICS[metric]()
    
# python metrics.py cosine
# python metrics.py jaccard