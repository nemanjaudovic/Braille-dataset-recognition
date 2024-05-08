import csv
import glob
from PIL import Image
import os
import numpy as np

direct_path = "Path/to/dataset"
jpg_files = sorted(glob.glob(os.path.join(direct_path, '*.jpg')))
a = 0
b=0
with open("FILE.txt", "w") as fajl:
    for f in jpg_files:
        image = Image.open(f)
        pixels = np.array(image)
        pixel_values = pixels.flatten()
        for p in pixel_values:
            fajl.write(str(p))
            fajl.write(" ")
        fajl.write(str(a))
        fajl.write('\n')
        b+=1
        if b==60:
            a+=1
            b=0






