from flask import Flask
from flask import render_template,jsonify,request
import random,string
import os,sys
app = Flask(__name__,static_folder="./uploads")

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

@app.route('/')
def index():
    return render_template('index.html', name='在线测试')

if __name__ == '__main__':
    app.run()
