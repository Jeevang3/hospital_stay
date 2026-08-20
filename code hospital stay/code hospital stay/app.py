#importing all required python libraries
import pandas as pd
#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics 
import seaborn as sns
import matplotlib.pyplot as plt #use to visualize dataset values
import os

app = Flask(__name__)
app.secret_key = 'welcome'

#loading and displaying hospital stay icu dataset
dataset = pd.read_csv("Dataset/LengthOfStay.csv", nrows=2000)
dataset

#describing dataset to know distribution of values ranges in each column
dataset.describe()

#dataset pre-processing like converting non-numeric data to numeric data using label encoder class
dataset['vdate'] = pd.to_datetime(dataset['vdate'])
dataset['year'] = dataset['vdate'].dt.year
dataset['month'] = dataset['vdate'].dt.month
dataset['day'] = dataset['vdate'].dt.day
label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for i in range(len(types)):
    name = types[i]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[i], le])
Y = dataset['lengthofstay'].ravel()
dataset.drop(['eid', 'vdate','lengthofstay'], axis = 1,inplace=True)#drop ir-relevant columns
print("Cleaned & Processed Dataset")
dataset

#applying imputation to replace missing values with mean
dataset = dataset.fillna(dataset.mean())
columns = dataset.columns
X = dataset.values
#normalizing training features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print("Normalized Training Features = "+str(X))

#split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print("Dataset Train & Test Split Details")
print("80% records used to train algorithms : "+str(X_train.shape[0]))
print("20% records used to test algorithms : "+str(X_test.shape[0]))
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

#define global variables to save accuracy and other metrics
accuracy = []
precision = []
recall = []
fscore = []
labels = ['Short Stay', 'Long Stay']

@app.route('/Predict', methods=['GET', 'POST'])
def predictView():
    return render_template('Predict.html', msg='')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')
    
@app.route('/index', methods=['GET', 'POST'])
def index1():
    return render_template('index.html', msg='')


@app.route('/AdminLogin', methods=['GET', 'POST'])
def AdminLogin():
    return render_template('AdminLogin.html', msg='')

@app.route('/AdminLoginAction', methods=['GET', 'POST'])
def AdminLoginAction():
    if request.method == 'POST' and 't1' in request.form and 't2' in request.form:
        user = request.form['t1']
        password = request.form['t2']
        if user == "admin" and password == "admin":
            return render_template('AdminScreen.html', msg="Welcome "+user)
        else:
            return render_template('AdminLogin.html', msg="Invalid login details")

@app.route('/Logout')
def Logout():
    return render_template('index.html', msg='')

def algorithm():
    global extension_model
    dataset = pd.read_csv("Dataset/LengthOfStay.csv", nrows=2000)
    #dataset pre-processing like converting non-numeric data to numeric data using label encoder class
    dataset['vdate'] = pd.to_datetime(dataset['vdate'])
    dataset['year'] = dataset['vdate'].dt.year
    dataset['month'] = dataset['vdate'].dt.month
    dataset['day'] = dataset['vdate'].dt.day
    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric
            label_encoder.append([columns[i], le])
    Y = dataset['lengthofstay'].ravel()
    dataset.drop(['eid', 'vdate','lengthofstay'], axis = 1,inplace=True)#drop ir-relevant columns
    print("Cleaned & Processed Dataset")
    #applying imputation to replace missing values with mean
    dataset = dataset.fillna(dataset.mean())
    columns = dataset.columns
    X = dataset.values
    #normalizing training features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print("Normalized Training Features = "+str(X))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)

    extension_model = CatBoostClassifier(iterations = 20)
    #training CatBoost algorithm on training features
    extension_model.fit(X_train, y_train)
    #perform prediction on test data
    predict = extension_model.predict(X_test)

algorithm()
@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        testData = pd.read_csv("Dataset/testData.csv")
        data = testData.values
        testData['vdate'] = pd.to_datetime(testData['vdate'])#convert date into numeric date format
        testData['year'] = testData['vdate'].dt.year
        testData['month'] = testData['vdate'].dt.month
        testData['day'] = testData['vdate'].dt.day
        for i in range(len(label_encoder)):#convert string data to numeric values
            le = label_encoder[i]
            testData[le[0]] = pd.Series(le[1].transform(testData[le[0]].astype(str)))#encode all str columns to numeric
        testData.drop(['eid', 'vdate'], axis = 1,inplace=True)#drop ir-relevant columns
        #handling missing values using imputation
        testData = testData.fillna(dataset.mean())
        testData = testData.values
        testData = scaler.transform(testData)#normalize test data
        predict = extension_model.predict(testData)#perform prediction on test data using extension model
        output = ""
        for i in range(len(predict)):
            output += "Test Data = "+str(data[i])+" Predicted ICU Stay ====> "+labels[predict[i]]+"<br/><br/>" 
        return render_template('AdminScreen.html', msg=output)

if __name__ == '__main__':
    app.run()

