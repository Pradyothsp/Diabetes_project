from flask import Flask, request, redirect, flash
import model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def do():
    html = '''
    <!doctype html>
    <title>DIABETES PROJECT</title>
    <h2>Please input parameters</h2>
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
        <input type=button value=submit />
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

        a = model.main(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        print(a)
        html = html + a


    return html

if __name__ == "__main__":
    app.run()