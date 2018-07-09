import os, sys
import utils.fingerprints as finger


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ("wrong number of arguments")
        print ("python .\metrics.py <fingerprints> <size>")
        sys.exit()

    fingerprints = sys.argv[1]
    size = sys.argv[2]
    filepath = './images/'+fingerprints

    new_folder = filepath+'_size'+str(size)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for dirpath, dnames, fnames in os.walk(filepath):
        for f in fnames:
            scaled_img = finger.scale_fingerprint(filepath+'/'+f, size)
            scaled_img.save(new_folder+'/'+f) 


# python scale_images.py SDSOM_64_5545_571698_id23 32
# python scale_images.py SDSOM_64_5545_571698_id23 50