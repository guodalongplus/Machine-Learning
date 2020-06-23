# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:14:20 2020

@author: dalong
"""
import pandas as pd
import numpy as np
from keras.layers import Input, LSTM, Dense, merge,concatenate,Embedding
from keras.optimizers import Adam, SGD
from keras.models import Model,load_model
from keras.utils import plot_model
from keras.models import Sequential
NUM_SAMPLES=3000
batch_size = 64
epochs = 300
latent_dim = 256 # LSTM 的单元个数
num_samples = 10000 # 训练样本的大小

data_path='data/cmn.txt'
df=pd.read_table(data_path,header=None).iloc[:NUM_SAMPLES,0:2]
df.columns=['inputs','targets']
#每句中文举手加上‘\t’作为起始标志，句末加上‘\n’终止标志
df['targets']=df['targets'].apply(lambda x:'\t'+x+'\n')

#获取英文、中文列表
input_texts=df.inputs.values.tolist()
target_texts=df.targets.values.tolist()

#确定中英文各自包含的字符。df.unique()直接取sum可将unique数组中的各个句子拼接成一个长句子
input_characters = sorted(list(set(df.inputs.unique().sum())))
target_characters = sorted(list(set(df.targets.unique().sum())))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
INUPT_LENGTH = max([ len(txt) for txt in input_texts])
OUTPUT_LENGTH = max([ len(txt) for txt in target_texts])
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length of input:', INUPT_LENGTH)
print('Max sequence length of outputs:', OUTPUT_LENGTH)

# 建立 字符->数字 字典，用于字符的向量化te(
input_token_index = dict( [(char, i)for i, char in enumerate(input_characters)] )
target_token_index = dict( [(char, i) for i, char in enumerate(target_characters)] )

#每条句子经过对字母转换成one-hot编码后，生成了LSTM需要的三维输入[n_samples, timestamp, one-hot feature]
encoder_input_data =np.zeros((NUM_SAMPLES,INUPT_LENGTH,num_encoder_tokens))
decoder_input_data =np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,num_decoder_tokens))
decoder_target_data  = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,num_decoder_tokens))

for i,(input_text,target_text) in enumerate(zip(input_texts,target_texts)):
    for t,char in enumerate(input_text):
        encoder_input_data[i,t,input_token_index[char]]=1.0

    for t, char in enumerate(target_text):
        decoder_input_data[i,t,target_token_index[char]]=1.0

        if t > 0:
            # decoder_target_data 不包含开始字符，并且比decoder_input_data提前一步
            decoder_target_data[i, t-1, target_token_index[char]] = 1.0

def create_model():
    #定义编码器的输入
    encoder_inputs=Input(shape=(None,num_encoder_tokens))
    #Embedding
    encoder_embedding_layer = Embedding(num_encoder_tokens, latent_dim)
    encoder_embedding=encoder_embedding_layer(encoder_inputs)
    #返回状态
    encoder=LSTM(latent_dim,return_state=True)
    # 调用编码器，得到编码器的输出（输入其实不需要），以及状态信息 state_h 和 state_c
    encoder_outputs,state_h,state_c=encoder(encoder_embedding)
    # 丢弃encoder_outputs, 我们只需要编码器的状态
    encoder_state=[state_h,state_c]
    
    
    #定义解码器的输入
    decoder_inputs=Input(shape=(None,num_decoder_tokens))
    decoder_embedding_layer = Embedding(num_decoder_tokens, latent_dim)
    decoder_embedding=decoder_embedding_layer(decoder_embedding_layer)
    # 并且返回其中间状态，中间状态在训练阶段不会用到，但是在推理阶段将是有用的
    decoder_lstm=LSTM(latent_dim,return_state=True,return_sequences=True)
    # 将编码器输出的状态作为初始解码器的初始状态
    decoder_outputs,_,_=decoder_lstm(decoder_embedding,initial_state=encoder_state)
    #添加全连接层
    decoder_dense=Dense(num_decoder_tokens,activation='softmax')
    decoder_outputs=decoder_dense(decoder_outputs)
    
    #定义整个模型
    model=Model([encoder_inputs,decoder_inputs],decoder_outputs)
    
    # 定义 sampling 模型
    # 定义 encoder 模型，得到输出encoder_states
    encoder_model=Model(encoder_inputs,encoder_state)
    
    decoder_state_input_h=Input(shape=(latent_dim,))
    decoder_state_input_c=Input(shape=(latent_dim,))
    decoder_state_inputs=[decoder_state_input_h,decoder_state_input_c]
    
    # 得到解码器的输出以及中间状态
    decoder_outputs,state_h,state_c=decoder_lstm(decoder_inputs,initial_state=decoder_state_inputs)
    decoder_states=[state_h,state_c]
    decoder_outputs=decoder_dense(decoder_outputs)
    decoder_model=Model([decoder_inputs]+decoder_state_inputs,[decoder_outputs]+decoder_states)
    
    plot_model(model=model,show_shapes=True)
    plot_model(model=encoder_model,show_shapes=True)
    plot_model(model=decoder_model,show_shapes=True)
    return model,encoder_model,decoder_model


# 建立 数字->字符 的字典，用于恢复
reverse_input_char_index = dict([(i, char) for char, i in input_token_index.items()])
reverse_target_char_index = dict([(i, char) for char, i in target_token_index.items()])

def decode_sequence(input_seq,encoder_model,decoder_model):
    # 将输入序列进行编码
    states_value = encoder_model.predict(input_seq)

    # 生成一个size=1的空序列
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # 将这个空序列的内容设置为开始字符
    target_seq[0, 0, target_token_index['\t']] = 1.

    # 进行字符恢复
    # 简单起见，假设batch_size = 1
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 退出条件：生成 \n 或者 超过最大序列长度
        if sampled_char == '\n' or len(decoded_sentence) >INUPT_LENGTH  :
            stop_condition = True

        # 更新target_seq
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # 更新中间状态
        states_value = [h, c]

    return decoded_sentence



# 检验成果的时候到了,从训练集中选取一些句子做测试
# 效果还行（废话，从训练集里挑的数据
#model,encoder_model,decoder_model=create_model()
##编译模型
#model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
###训练模型
#model.fit([encoder_input_data,decoder_input_data],decoder_target_data,
#          batch_size=batch_size,
#          epochs=epochs,
#          validation_split=0.2)
#model.save('s2s.h5')
#encoder_model.save('encoder_model.h5')
#decoder_model.save('decoder_model.h5')
   
encoder_model=load_model('encoder_model.h5') 
decoder_model=load_model('decoder_model.h5') 
def test():
    ss=input("请输入要翻译的英文:")
    if ss=="-1":
        exit(-1)
    input_seq=np.zeros((1,INUPT_LENGTH,num_encoder_tokens)) 
    for t,char in enumerate(ss):
        input_seq[0,t,input_token_index[char]]=1.0
    decoded_sentence = decode_sequence(input_seq,encoder_model,decoder_model)
    print('-')
    print('Decoded sentence:', decoded_sentence)

while(1):
    test()















