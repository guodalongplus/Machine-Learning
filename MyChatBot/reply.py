# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 08:48:06 2020

@author: dalong
"""

import jieba
import pickle
import numpy as np
from keras.models import load_model

INPUT_LENGTH=13
num_encoder_symbols=7885
class ChatbotRespnse():
    words = pickle.load(open('../pickle_and_h5/words.pkl','rb'))
    word_index = pickle.load(open('../pickle_and_h5/word_index.pkl','rb'))
    index_word = pickle.load(open('../pickle_and_h5/index_word.pkl','rb'))
    encoder_model=load_model('../pickle_and_h5/encoder_model.h5', compile=False)
    decoder_model=load_model('../pickle_and_h5/decoder_model.h5', compile=False)
    
    def response(self,source,encoder_inference, decoder_inference):
        states_value = encoder_inference.predict(source)
    
        # 生成一个size=1的空序列
        target_seq = np.zeros((1, 1, num_encoder_symbols))
        # 将这个空序列的内容设置为开始字符
        target_seq[0, 0, self.word_index['START']] = 1.
    
        # 进行字符恢复
        # 简单起见，假设batch_size = 1
        stop_condition = False
        decoded_sentence = ''
    
        while not stop_condition:
            output_tokens, h, c = decoder_inference.predict([target_seq] + states_value)
    #        print(output_tokens)
    
            # sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.index_word[sampled_token_index]
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
    
    def chatbot_response(self,ss):
    
        input_seq=np.zeros((1,INPUT_LENGTH,num_encoder_symbols)) 
        ss_seg=jieba.lcut(ss)
        for t,char in enumerate(ss_seg):
            try:
                input_seq[0,t,self.word_index[char]]=1.0
            except:
                return "你说啥呢？我的脑洞没那么大~~~百度去吧"
        decoded_sentence = self.response(input_seq,self.encoder_model,self.decoder_model)
                
        return decoded_sentence