import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Suggestion: Heatmap

header_list = ["Class", "Detection", "Classification"]
df = pd.read_csv("detection_scores.txt", delimiter ="|", names= header_list)

sum = 0

print((df["Classification"]>=0.9).count())
# df = df.T

# # Making class names as column headers
# new_header = df.iloc[0] #grab the first row for the header
# df = df[1:] #take the data less the header row
# df.columns = new_header #set the header row as the df header

# #Groupby class names
# label_encoder = LabelEncoder()
# label_encoder.fit_transform(df.columns)
# classes = label_encoder.classes_


# df=df.groupby(by=classes)
# print(df.head)



# plt.matshow(df.corr())
# plt.clim(0.0,1.0) 
# plt.colorbar()
# plt.savefig('detection_classification.png')

# fig = figure(figsize=(20, 20), dpi=80)
# sns.heatmap(df.cov(), annot=True)
# plt.savefig('detection_classification.png')

# scores = np.asarray(list(zip(detection_scores, classification_scores)))
# print(scores.shape)

# # detection_scores = np.asarray(detection_scores)
# # detection_scores = detection_scores.reshape((-1,1))

# # classification_scores = np.asarray(classification_scores)
# # classification_scores = classification_scores.reshape((-1,1))



#detection_scores =  np.asarray(list(zip(classes, detection_scores)))


# fig = figure(figsize=(20, 20), dpi=80)
# ax = sns.heatmap(classes, label="detection")
# # ax = sns.heatmap(classification_scores, label="classification")
# plt.savefig('detection_classification.png')


fig = figure(figsize=(15, 15), dpi=80)
ax1 = fig.add_subplot(111)
ax1.scatter(df["Classification"], df["Class"], label="classification")
ax1.scatter(df["Detection"],df["Class"], label="detection")

plt.legend(loc='upper left')
plt.show()
plt.savefig('detection_classification.png')


# f = open('inference_results.json')
# class_dict = json.load(f)

# classification_scores = []
# classes = []
# for key in class_dict:
#     #print(class_dict[key][0]['classification_scores'][0])
#     classification_scores.append(class_dict[key][0]['classification_scores'][0])
#     classes.append(class_dict[key][0]['classification_name_ids'][0])
    
# #plot

# pyplot.scatter(classification_scores,classes)
# pyplot.show()
# pyplot.savefig('classification_confidence.png')
  
