from flask import Flask, request, redirect, flash, send_file
import model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def do():
    html = '''
    <!doctype html>
    <title>DIABETES PREDICTION</title>
    <h2>Please Input Parameters of the Person</h2>
    <style>
    body {
        background-image: url('background_image/doctor.jpg');
        background-repeat: no-repeat;
        background-attachment: fixed; 
        background-size: 100% 100%;
    }
    </style>
    <body>
    <form method=POST enctype=multipart/form-data action='/'>
        <label for="field1">Pregnancies:</label>
        <input type="text" id="field1" name="Pregnancies" /><br><br>
        <label for="field2">Glucose:</label>
        <input type="text" id="field2" name="Glucose" /><br><br>
        <label for="field3">BloodPressure:</label>
        <input type="text" id="field3" name="BloodPressure" /><br><br>
        <label for="field4">SkinThickness:</label>
        <input type="text" id="field4" name="SkinThickness" /><br><br>
        <label for="field5">Insulin:</label>
        <input type="text" id="field5" name="Insulin" /><br><br>
        <label for="field6">BMI:</label>
        <input type="text" id="field6" name="BMI" /><br><br>
        <label for="field7">DiabetesPedigreeFunction:</label>
        <input type="text" id="field7" name="DiabetesPedigreeFunction" /><br><br>
        <label for="field8">Age:</label>
        <input type="text" id="field8" name="Age" /><br><br>
        <input type=submit value=submit />
    </body>
    </form>
    '''

    if request.method == 'POST':
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])

        knn_, logreg_, tree_, rf_, gb_, svc_ = model.main(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        
        html = html + knn_ + '<br />' + logreg_ + '<br />' + tree_ + '<br />' + rf_ + '<br />' + gb_ + '<br />' + svc_
        
    return html

@app.route('/background_image/<name>')
def run_detector(name):
    return send_file('background_image/'+name)


if __name__ == "__main__":
    app.run()