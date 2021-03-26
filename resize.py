#!/usr/bin/python
from PIL import Image
import os, sys

dir = "/ocean/projects/pscstaff/rajanie/MLStamps/long-tail/OpenLongTailRecognition-OLTR/OLTRDataset/OLTRDataset_1/test_inference"
out = '/ocean/projects/pscstaff/rajanie/MLStamps/long-tail/OpenLongTailRecognition-OLTR/OLTRDataset/OLTRDataset_1/test_inference_256x256'

def resize():
    count = 0
    for i in os.listdir(dir):
    #files = 0
        #for file in os.listdir(os.path.join(dir,i)):
            #if os.path.isfile(path+item):
        im = Image.open(dir+"/"+i)
        print(i)
        print(count)
        count += 1
               # f, e = os.path.splitext(path+item)
        imResize = im.resize((256,256), Image.BILINEAR)
        imResize.save(out+"/"+i, 'JPEG', quality=90)

resize()
