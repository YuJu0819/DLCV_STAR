import json
import csv
import os
#read csv
os.listdir("./result")
ans = {}
for file in os.listdir("./result"):
    with open('./result/'+file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    for i in range(1,len(data)):
        if data[i][0] not in ans.keys():
            ans[data[i][0]] = [0,0,0,0]#0123
            ans[data[i][0]][int(data[i][1])] += 1
        else:
            ans[data[i][0]][int(data[i][1])] += 1
for i in range(1,len(ans)+1):
    ans[data[i][0]] = ans[data[i][0]].index(max(ans[data[i][0]]))
# print(ans[data[1][0]])
aans = []
for i in range(1,len(ans)+1):
    aans.append( [ data[i][0], ans[data[i][0]] ] )
# print(aans)

with open('submission_example.json', 'r') as f:
    sample = json.load(f)

ans = {"Interaction":[], "Sequence":[], "Prediction":[], "Feasibility":[]}
data = aans
# data = data1 + data2 + data3

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

with open('vqa_vote_new_wo_tmp.json', 'w') as f:
    json.dump(ans, f)