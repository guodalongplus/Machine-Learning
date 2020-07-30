demo_data:      不同大小的语料信息
pic:            神经网络结构图片
pickle_and_h5:  序列化的语料信息训练好的模型
chatbot:        训练模型/cmd模式测试模型
ui/app.py       flask部署下的网页交互模式

requrements:
keras       '2.3.1'
tensorflow  '2.1.0'


具体操作流程：
1.将问答句用jieba进行分词，根据词频排序
2.将词库转化为字典形式(index-word和word-index),并序列化
3.把句子长度大于某个值的句子过滤掉，并在回答句子的开头加上START,末尾加上EOS标识
4.(one_hot_encoder)将问答句进行编码 注意目标序列需要和回答的句子相差一个时间步长
5.(create_model)搭建模型
6.(response)因为我们的模型的输入和输出都是向量，所以需要对输入的句子进行向量化和对输出的结果进行反推找到它对应的文字。
7.可以在cmd测试效果或者在ui目录下的app.py在网页访问



