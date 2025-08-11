from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=int(request.form.get('age')),
            workclass=request.form.get('workclass'),
            fnlwgt=int(request.form.get('fnlwgt')),
            education=request.form.get('education'),
            education_num=int(request.form.get('education_num')),
            marital_status=request.form.get('marital_status'),
            occupation=request.form.get('occupation'),
            relationship=request.form.get('relationship'),
            race=request.form.get('race'),
            sex=request.form.get('sex'),
            capital_gain=int(request.form.get('capital_gain')),
            capital_loss=int(request.form.get('capital_loss')),
            hours_per_week=int(request.form.get('hours_per_week')),
            native_country=request.form.get('native_country')
)
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        


