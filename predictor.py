#imports for fileupload
#imports for fileupload
from glob import glob
from imp import init_frozen
from operator import methodcaller
from flask import Flask, jsonify, request, Response, redirect, json
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
#imports for ml purpose
import json
import pandas as pd
import matplotlib
import numpy as np

import io
from datetime import datetime
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import statsmodels.api as sm
import itertools




#rcparams for trend, seasonality, noise graph
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")


#created a flask application
app = Flask(__name__)
CORS(app)
global stepCount
#file upload - getting post request from angular-flask
app.config['UPLOAD_FOLDER'] = 'D:\salesforeml-main\salesforeml-main'
@app.route('/upload',methods = ['GET', 'POST'])
def upload_File():

    if request.method == 'POST':
        #check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        global fname
        fname = secure_filename(file.filename) 
        print(fname)   
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
        
        return redirect(request.url)

    return 'File Uploaded'


tag ='=====================>'
@app.route('/forecast', methods = ['POST','GET'])
def forecast():
    info = request.data
    #print(tag,info)
    dict_str = str(info, 'UTF-8')
    #print(tag,dict_str)
    data = json.loads(dict_str)
    steps = data['selectedItem']
    # print(tag,dict)
    # print(dict.keys())

    # steps = dict.get('period')
    # print(steps)
    global stepCount
    stepCount =int(steps)
    print(stepCount)
    
    return 'success!'
#def matplotlib_pyplot_savefig():
    #plt.savefig('plot.png')



#create a parser to modify the stepCount format
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

#show the final graph required
length = 14
breadth = 7
@app.route('/plot',methods = ['GET', 'POST'])
def plot():
    # Load the data from CSV
    plt.clf()
    data = pd.read_csv(fname, index_col='Date', parse_dates=['Date'])
    filename = "Prediction.xlsx"

# construct the file path using the current working directory
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"The file '{filename}' has been deleted.")
    else:
        print(f"The file '{filename}' does not exist.")
    # os.remove(file_path)

    # Select the endogenous variable
    endog = data['Sales']

    # Define the SARIMAX model
    model = SARIMAX(endog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

    # Fit the model
    results = model.fit()
    # first_column = data.iloc[:, 0]
    # print(first_column)
    # data.columns = ['Date']
    # Generate a forecast for the specified number of steps
    forecast = results.forecast(steps=stepCount)
    fore=forecast.to_frame()
    fore.reset_index(inplace=True)
    fore.rename(columns={'index': 'date'}, inplace=True)

# Convert date column to datetime format
    fore['date'] = pd.to_datetime(fore['date'], unit='s')
    fore['date'] = fore['date'].dt.strftime('%d-%m-%Y')
    # fore.columns = ['Date']
    fore.columns = ['Date','Prediction']
    print(type(forecast))
    print("Forecast")
    print(fore)
    # forecast_df = forecast.tolist()
    # print(forecast_df)
    
    # Save the forecast dataframe to Excel
    
    fore.to_excel('Prediction.xlsx')

# Save the DataFrame to Excel
    
    plt.rcParams['figure.figsize'] = [14, 7]
    plt.plot(endog.index, endog, label='Actual Sales')
    plt.plot(forecast.index, forecast, label='Forecast')
    if(stepCount!=60):
        for i, j in zip(endog.index, endog):
            plt.text(i, j, str(j))
    # for k, l in zip(forecast.index, forecast):
    #     plt.text(k, l, str(k))
    
    plt.legend()
    plt.savefig('my_plot.png', bbox_inches='tight')
   # data.to_excel('Prediction.xlsx')
    plt.show()
    forecast = forecast.tolist()
    
    

    # Return the forecast as a JSON response
    return jsonify(forecast=forecast)


# @app.route('/create',methods=['GET','POST'])
# def create_excel():
#     data1.to_excel('Prediction.xlsx')

#     return redirect("http://localhost:4200/forecast")
# A simple function to calculate the square of a number
@app.route('/home/<int:num>', methods = ['GET'])
def disp(num):
    return jsonify({'data': num**2})


# driver function
if __name__ == '__main__':
    app.run(debug = True)