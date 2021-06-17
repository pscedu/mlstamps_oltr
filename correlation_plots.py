import csv
import json

with open ('detection_scores.txt', 'r') as f:
    detect_conf = [row for row in csv.reader(f,delimiter='|')]

print (detect_conf[0][0])   
print (detect_conf[0][1])   
print (detect_conf[0][2])    


f = open('inference_results.json')
class_dict = json.load(f)
        
print("Printing each JSON Decoded Object")
for c in class_dict:
    print(c)
    break
  
