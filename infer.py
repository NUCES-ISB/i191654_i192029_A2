from flask import Flask, render_template, request
import numpy
import pickle
from sklearn import preprocessing, svm

app = Flask(__name__,template_folder='templates') #creating the Flask class object   
 
def getMetrics():
    with open("model.pkl","rb") as f:
        loaded_model=pickle.load(f)
    x=[[134.8  ],
 [134.45 ],
 [125.7  ],
 [145.935],
 [132.52 ],
 [131.19 ],
 [126.6  ],
 [121.65 ],
 [132.44 ],
 [143.75 ],
 [121.69 ]]
    y=[136.33, 133.41 ,125.57, 146.8,  131.24, 127.85, 126.74, 122.15, 134.43, 142.45,
 119.98 ]
    re=loaded_model.score(x,y)
    return str(re)
def getAccuracy():
    with open("model.pkl","rb") as f:
        loaded_model=pickle.load(f)
    return loaded_model.intercept_
def getResults(num):
    # save the model to disk
    num1=int(num)
    arr=numpy.array(num,dtype=float)
 
    num2=arr.reshape(1,-1)
    print(num2)
#C:\\Users\\sillah\\Documents\\Tweet\\model\\
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    value= loaded_model.predict(num2)
    print(value[0])
    return str(value[0])

@app.route('/', methods =["GET", "POST"])
def home():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       years = request.form.get("open")
       predictedSal=getResults(years)
       intercept=getAccuracy()
       metrics=getMetrics()
       return "Predicted closed value for apple share is: "+predictedSal+" and the intercept of model is: "+str(intercept)+" and the accuracy is : "+metrics
   
    return render_template("home.html")
  
if __name__ =='__main__':  
    app.run(debug = True,host='0.0.0.0',port=5000)  
    
# # start flask
# app = Flask(__name__)

# # render default webpage
# @app.route('/')
# def home():
#     if request.method == 'POST':
#         if request.form.get('action1') == 'VALUE1':
#             print("yeah")
#     elif request.method == 'GET':
#         return render_template('home.html', form=form)
    
#     return render_template('home.html')

# # when the post method detect, then redirect to success function
# @app.route('/', methods=['POST', 'GET'])
# def get_data():
#     if request.method == 'POST':
#         user = request.form['search']
#         return redirect(url_for('success', name=user))

# # get the data for the requested query
# @app.route('/success/<name>')
# def success(name):
#     return "<xmp>" + str(requestResults(name)) + " </xmp> "
