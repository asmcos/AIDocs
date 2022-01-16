from flask import Flask
from flask import render_template,jsonify,request
import random,string
import os
app = Flask(__name__,static_folder="./uploads")


if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route("/uploads",methods=['GET', 'POST'])
def uploadsfile():
    if request.method == 'POST':
        f = request.files['image']
        randval = "".join(random.sample(string.digits+string.ascii_letters,8))
        f.save('uploads/'+randval+f.filename)

    return jsonify({"errno": 0,\
             "data": ["/uploads/"+randval+f.filename,]
            })

@app.route('/')
def index():
    return render_template('index.html', name='在线测试')

if __name__ == '__main__':
    app.run()
