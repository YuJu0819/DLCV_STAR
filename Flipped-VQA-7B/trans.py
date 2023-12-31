import json
import csv

#read csv
with open('result.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
with open('submission_example.json', 'r') as f:
    sample = json.load(f)

ans = {"Interaction":[], "Sequence":[], "Prediction":[], "Feasibility":[]}

for i in range(len(sample["Interaction"])):
    for j in range(len(data)):
        if sample["Interaction"][i]["question_id"] == data[j][0]:
            ans["Interaction"].append({"question_id":data[j][0], "answer":int(data[j][1])})
for i in range(len(sample["Sequence"])):
    for j in range(len(data)):
        if sample["Sequence"][i]["question_id"] == data[j][0]:
            ans["Sequence"].append({"question_id":data[j][0], "answer":int(data[j][1])})
for i in range(len(sample["Prediction"])):
    for j in range(len(data)):
        if sample["Prediction"][i]["question_id"] == data[j][0]:
            ans["Prediction"].append({"question_id":data[j][0], "answer":int(data[j][1])})
for i in range(len(sample["Feasibility"])):
    for j in range(len(data)):
        if sample["Feasibility"][i]["question_id"] == data[j][0]:
            ans["Feasibility"].append({"question_id":data[j][0], "answer":int(data[j][1])})

print(len(ans["Interaction"]))
print(len(ans["Sequence"]))
print(len(ans["Prediction"]))
print(len(ans["Feasibility"]))

with open('vqa_submission.json', 'w') as f:
    json.dump(ans, f)
