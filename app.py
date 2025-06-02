from __future__ import division, print_function
# coding=utf-8
import sys
import os
import os
from PIL import Image
import glob
import re
import numpy as np
import cv2
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import sqlite3
import pandas as pd
import numpy as np
import pickle
import sqlite3
import random

import smtplib 
from email.message import EmailMessage
from datetime import datetime



app = Flask(__name__)

labels = ['Mono', 'Poly']

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_path2 = 'model.h5' # load .h5 Model


CTS = load_model(model_path2)
    
   
@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')



@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/predict2',methods=['GET','POST'])
def predict2():
    if request.method == "POST":
         
        print("Entered")
        
        print("Entered here")
        file = request.files['file'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)


            
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        img = cv2.imread(file_path)#read test image
        img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_CUBIC) #scale imaage
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#hsv transform
        img = cv2.flip(img, 1)#flip images
        img = img.reshape(1,32,32,3)#convert image as 4 dimension
        predict = CTS.predict(img)#predict solar defect from test image
        predict_label = predict[1] #get classification defect labels
        defect_probability = predict[0][0][0]#get defect probability
        predict_label = np.argmax(predict_label)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (600,400))#display image with predicted output
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, 'Predicted As : '+labels[predict_label], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.putText(img, 'Defect Probability : '+str(defect_probability), (10, 65),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        img_base64 = Image.fromarray(img)

        img_base64.save("static/image0.jpg", format="JPEG")

        return redirect("static/image0.jpg")
              
    return render_template('index.html')



@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "vandhanatruprojects@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("vandhanatruprojects@gmail.com", "pahksvxachlnoopc")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict1', methods=['POST'])
def predict1():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signin.html")

@app.route("/notebook")
def notebook1():
    return render_template("SolarDefectDetection.html")


   
if __name__ == '__main__':
    app.run(debug=False)