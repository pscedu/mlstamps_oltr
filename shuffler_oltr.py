import sys
sys.path.append('/ocean/projects/pscstaff/rajanie/MLStamps/long-tail/OpenLongTailRecognition-OLTR')
from interface.interfacePytorch import ObjectDataset


db_file = '/ocean/projects/pscstaff/rajanie/MLStamps/long-tail/OpenLongTailRecognition-OLTR/campaign3to6-6Kx4K.v4-stamp.db'
rootdir = '/ocean/projects/pscstaff/rajanie/MLStamps/long-tail/OpenLongTailRecognition-OLTR/data'
dataset = ObjectDataset(db_file, rootdir=rootdir)
item = dataset.__getitem__(1)
#print(pformat(item))
print((item))