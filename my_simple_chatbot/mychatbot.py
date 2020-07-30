import json
import pickle
import jieba
import numpy as np
import random
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#%% Part 1 -语料预处理
ignore_words=["_","@","-","~"]
#with open("QA.json","rb",encoding="utf-8") as f:
#    intents=json.load(f)
datafile=open("tinyQA.json",'rb').read()
intents=json.loads(datafile.decode())
classes=[]
total_words=[]
documents=[]

for q_a in intents['intents']:
    w=jieba.lcut(q_a['patterns'])
    total_words.extend(w)
    if q_a['tag'] not in classes:
        classes.append(q_a['tag'])
        documents.append((w, q_a['tag']))
clean_words = sorted(set([w for w in total_words if w not in ignore_words]))
classes=sorted(set(classes))

print(len(documents), "documents")
print(len(classes), "classes")
print(len(clean_words), "clean_words")
#将classes和分析结果序列化
pickle.dump(clean_words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#%% Part 2 -create trainingset
training=[]
iloc_class=[0]*len(classes)
for one_round in documents:
    bags=[]
    pattern_words=one_round[0]
    for i in clean_words:
        bags.append(1) if i in pattern_words else bags.append(0)
    newclass=list(iloc_class)    
    newclass[classes.index(one_round[1])] = 1
    training.append([bags, newclass])
    
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")
#%% create model
starttime=time.time()
print(starttime)
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=20000, batch_size=10, verbose=1)
model.save('CH_chatbot_model.h5', hist)
endtime=time.time()
t=endtime-starttime
print("endtime: ",endtime)
print("用时：",str(t/3600),"小时")
print("model created")