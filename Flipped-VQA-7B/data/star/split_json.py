import json

with open('STAR_test.json') as f:
    data = json.load(f)
    data1 = data[:len(data)//3]
    data2 = data[len(data)//3:2*len(data)//3]
    data3 = data[2*len(data)//3:]
    with open('STAR_test1.json', 'w') as f1:
        json.dump(data1, f1)
    with open('STAR_test2.json', 'w') as f2:
        json.dump(data2, f2)
    with open('STAR_test3.json', 'w') as f3:
        json.dump(data3, f3)