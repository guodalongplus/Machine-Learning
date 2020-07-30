# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:39:46 2020

@author: dalong
"""
import sys
import numpy as np
import jieba
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.optimizers import Adam, SGD
from keras.models import Model,load_model

from keras.utils.vis_utils import plot_model

import pickle


N_UNITS = 256
BATCH_SIZE = 64
EPOCH = 100
min_freq=2
NUM_SAMPLES=0
file_list=['./demo_data/q1.txt', './demo_data/a1.txt']
words_count = {}
for file in file_list:
    with open(file, 'rb') as file_object:
        for line in file_object.readlines():
            NUM_SAMPLES+=1
            line = line.strip()
            seg_list = jieba.cut(line)
            for s in seg_list:
                if s in words_count:
                     words_count[s]+=1
                else:
                     words_count[s]=1

sorted_list = [[v[1], v[0]] for v in words_count.items()]
sorted_list.sort(reverse=True)
words=[]
for index, item in enumerate(sorted_list):
    word = item[1]
    if item[0] < 0:
        continue
    words.append(word)
words.append("EOS")   
words.append("START")   
pickle.dump(words,open('words.pkl','wb'))

word_index = dict( [(char, i)for i, char in enumerate(words)] )
index_word = dict([(i, char) for i, char in enumerate(words)])
pickle.dump(word_index,open('word_index.pkl','wb'))
pickle.dump(index_word,open('index_word.pkl','wb'))

num_encoder_symbols=len(words)#获取序列大小

input_texts=[]
output_texts=[]
NUM_SAMPLES=int(NUM_SAMPLES/2)
with open('./demo_data/q1.txt', 'r', encoding='UTF-8') as question_file:
    with open('./demo_data/a1.txt', 'r', encoding='UTF-8') as answer_file:
        for i in range (NUM_SAMPLES):
            question = question_file.readline()
            answer = answer_file.readline()
            if(len(question)>15 or len(answer)>15):
                continue
            if question and answer:
                question = question.strip()
                answer =answer.strip()
                input_seg=jieba.lcut(question)
                output_seg=["START"]+jieba.lcut(answer)+["EOS"]
                input_texts.append(input_seg)
                output_texts.append(output_seg)
            else:
                break
INPUT_LENGTH = max([len(list(i)) for i in input_texts])
OUTPUT_LENGTH = max([len(list(i)) for i in output_texts])


#将问答进行编码
def one_hot_encoder():
    encoder_input = np.zeros((NUM_SAMPLES,INPUT_LENGTH,num_encoder_symbols))
    decoder_input = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,num_encoder_symbols))
    decoder_output = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,num_encoder_symbols))

    for i, (input_text,target_text) in enumerate(zip(input_texts,output_texts)):
        for t,char in enumerate(input_text):
            encoder_input[i,t,word_index[char]]=1.0
            
        for t, char in enumerate(target_text):
            decoder_input[i,t,word_index[char]]=1.0
            
            if t > 0:
                # decoder_target_data 不包含开始字符，并且比decoder_input_data提前一步
                decoder_output[i, t-1, word_index[char]] = 1.0
    return encoder_input,decoder_input,decoder_output

def create_model(n_units):
    #训练阶段
    #encoder
    encoder_input = Input(shape = (None, num_encoder_symbols))
    #encoder输入维度n_input为每个时间步的输入xt的维度，这里是用来one-hot的英文字符数
    encoder = LSTM(n_units, return_state=True)
    #n_units为LSTM单元中每个门的神经元的个数，return_state设为True时才会返回最后时刻的状态h,c
    _,encoder_h,encoder_c = encoder(encoder_input)
    encoder_state = [encoder_h,encoder_c]
    #保留下来encoder的末状态作为decoder的初始状态
    
    #decoder
    decoder_input = Input(shape = (None, num_encoder_symbols))
    #decoder的输入维度为中文字符数
    decoder = LSTM(n_units,return_sequences=True, return_state=True)
    #训练模型时需要decoder的输出序列来与结果对比优化，故return_sequences也要设为True
    decoder_output, _, _ = decoder(decoder_input,initial_state=encoder_state)
    #在训练阶段只需要用到decoder的输出序列，不需要用最终状态h.c
    decoder_dense = Dense(num_encoder_symbols,activation='softmax')
    decoder_output = decoder_dense(decoder_output)
    #输出序列经过全连接层得到结果
    
    #生成的训练模型
    model = Model([encoder_input,decoder_input],decoder_output)
    #第一个参数为训练模型的输入，包含了encoder和decoder的输入，第二个参数为模型的输出，包含了decoder的输出
    
    #推理阶段，用于预测过程
    #推断模型—encoder
    encoder_infer = Model(encoder_input,encoder_state)
    
    #推断模型-decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))    
    decoder_state_input = [decoder_state_input_h, decoder_state_input_c]#上个时刻的状态h,c   
    
    decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,initial_state=decoder_state_input)
    decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]#当前时刻得到的状态
    decoder_infer_output = decoder_dense(decoder_infer_output)#当前时刻的输出
    decoder_infer = Model([decoder_input]+decoder_state_input,[decoder_infer_output]+decoder_infer_state)
    
    return model, encoder_infer, decoder_infer

#回答函数
def response(source,encoder_inference, decoder_inference):
    states_value = encoder_inference.predict(source)

    # 生成一个size=1的空序列
    target_seq = np.zeros((1, 1, num_encoder_symbols))
    # 将这个空序列的内容设置为开始字符
    target_seq[0, 0, word_index['START']] = 1.

    # 进行字符恢复
    # 简单起见，假设batch_size = 1
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_inference.predict([target_seq] + states_value)

        # sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_word[sampled_token_index]
        decoded_sentence += sampled_char

        # 退出条件：生成 EOS 或者 超过最大序列长度
        if sampled_char == 'EOS' or len(decoded_sentence) >INPUT_LENGTH  :
            stop_condition = True

        # 更新target_seq
        target_seq = np.zeros((1, 1, num_encoder_symbols))
        target_seq[0, 0, sampled_token_index] = 1.

        # 更新中间状态
        states_value = [h, c]

    return decoded_sentence[:-3]



def train():
    encoder_input_data,decoder_input_data,decoder_output=one_hot_encoder()
    model,encoder_model,decoder_model=create_model(N_UNITS)
    #编译模型
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
    ##训练模型
    model.fit([encoder_input_data,decoder_input_data],decoder_output,
              batch_size=BATCH_SIZE,
              epochs=EPOCH,
              validation_split=0.2)
    model.save('model.h5')
    encoder_model.save('./pickle_and_h5/encoder_model.h5')
    decoder_model.save('./pickle_and_h5/decoder_model.h5')
   

def test():
    encoder_model=load_model('./pickle_and_h5/encoder_model.h5', compile=False)
    decoder_model=load_model('./pickle_and_h5/decoder_model.h5', compile=False)
    ss=input("我:")
    if ss=='-1':
        sys.exit()
    input_seq=np.zeros((1,INPUT_LENGTH,num_encoder_symbols)) 
    ss_seg=jieba.lcut(ss)
    for t,char in enumerate(ss_seg):
        input_seq[0,t,word_index[char]]=1.0
    decoded_sentence = response(input_seq,encoder_model,decoder_model)
    print('-')
    print('robot:', decoded_sentence)

if __name__ == '__main__':
   intro=input("select train model or test model:")
   if intro=="train":
       print("训练模式...........")
       train()
   else:
       print("测试模式.........")
       while(1):
           test()



