from flask import Flask, render_template, request
import reply
from flask import jsonify
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

chatbot=reply.ChatbotRespnse()
app = Flask(__name__)

#############
# Routing

@app.route('/message', methods=['POST'])
def reply():
    print(request.form['msg'])
    return  jsonify( { 'text':chatbot.chatbot_response(request.form['msg'])})

@app.route("/")
def index():

    return render_template("index.html")





#_________________________________________________________________

# start app
if (__name__ == "__main__"):
    app.run(port = 4100)
    print("app:", chatbot.chatbot_response(str(request.form['msg'])))
