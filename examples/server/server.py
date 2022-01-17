from flask import Flask
from flask import render_template,jsonify,request
import random,string
import os,sys
from db import insert, get_all_pred,makeImagefrompred 

from dl_lession_week2 import pred_cat

app = Flask(__name__,static_folder="./uploads")

app.config['DEBUG'] = True

if not os.path.exists(sys.path[0]+'/uploads'):
    os.makedirs(sys.path[0]+'/uploads')

@app.route("/uploads",methods=['GET', 'POST'])
def uploadsfile():
    if request.method == 'POST':
        f = request.files['image']
        randval = "".join(random.sample(string.digits+string.ascii_letters,8))
        f.save(sys.path[0]+'/uploads/'+randval+f.filename)

    return jsonify({"errno": 0,\
             "data": ["/uploads/"+randval+f.filename,]
            })


@app.route("/pred",methods=['GET', 'POST'])
def pred():
    if request.method == 'POST':
        filename = request.form.get('filename')
        print(filename)
        print(pred_cat(sys.path[0]+filename))
        #insert(filename,pred)
    return jsonify({"pred":1})

@app.route("/accept",methods=['GET', 'POST'])
def accept():
    if request.method == 'POST':
        pass
        #makeImagefrompred(predid,label)
    return jsonify({"success":1})


@app.route('/')
def index():
    return render_template('index.html', name='在线测试')

if __name__ == '__main__':
    app.run()
