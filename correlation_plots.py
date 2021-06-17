import csv
import json

with open ('detection_scores.txt', 'r') as f:
    detect_conf = [row for row in csv.reader(f,delimiter='|')]

print (detect_conf[0][0])   
print (detect_conf[0][1])   
print (detect_conf[0][2])    


classification_scores = []
print("Started Reading JSON file which contains multiple JSON document")
with open('inference_results.json') as f:
    for jsonObj in f:
        class_dict = json.loads(jsonObj)
        classification_scores.append(class_dict)

print("Printing each JSON Decoded Object")
for c in classification_scores:
    print(c)
    break
  
