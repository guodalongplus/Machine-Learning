# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:59:21 2020

@author: dalong
"""

import json
import numpy as np

 
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
 

data={"intents":[]}
with open("./data/question.txt",'r', encoding='utf-8') as q:
    with open("./data/answer.txt",'r', encoding='utf-8') as a:
        i=1
        while True:
            Q=q.readline()
            A=a.readline()
            if Q and A:
                Q=Q.strip()
                A=A.strip()
                doc={}
                doc["tag"]=str(i)
                doc["patterns"]=Q
                doc["responses"]=A
                i=i+1
                data["intents"].append(doc)
            else:
                break
newdata=json.dumps(data,ensure_ascii=False)
with open('tinyQA.json','w',encoding='utf-8') as j:
    j.write(newdata)
