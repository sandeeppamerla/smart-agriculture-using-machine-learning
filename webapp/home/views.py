from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import keras
import tensorflow
from keras.layers import *
from .models import *


# Create your views here.
def index(request):
    return render(request, 'index.html')
# Uploading the dataset.


def upload_dataset(request):
    global data
    if request.method == 'POST':
        file = request.FILES['file']
        d = dataset(file=file)
        # d.save()
        fn = d.filename()
        print(fn)
        global data, path
        path = 'home/static/home/dataset/'+fn
        data = pd.read_csv('home/static/home/dataset/'+fn)
        datas = data.iloc[:100, :]
        table = datas.to_html()
        return render(request, 'upload.html', {'table': table})
    return render(request, 'upload.html')
# Training the data by using algorithms.


def train(request):
    global data
    data = pd.read_csv(
        'home/static/home/dataset/Crop_recommendation_Dd0sgfN.csv')
    # numerical_data = data.select_dtypes(include='number')
    # categorical_data = data.select_dtypes(exclude='number')
    # print(categorical_data.shape)
    # enc = OrdinalEncoder()
    # enc_data = enc.fit_transform(categorical_data)
    # data_enc = pd.DataFrame(enc_data, columns=categorical_data.columns)
    # data = pd.concat([data_enc, numerical_data], axis=1)
    # data.head()
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()

    global x_train, x_test, y_train, y_test
    if request.method == "POST":
        x = data.drop(['label'], axis=1)
        y = data['label']
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42)
        # print(y_train)
        print(x_train)
        print(y_train)
        model = request.POST['algo']

        if model == "1":
            clf = DecisionTreeClassifier()
            clf.fit(x_train._get_numeric_data(), np.ravel(y_train, order='C'))
            y_pred = clf.predict(x_test)
            clf = accuracy_score(y_test, y_pred)
            msg = 'Accuracy of DecisionTree : ' + str(clf*100)
            return render(request, 'train.html', {'acc': msg})

        elif model == "2":
            # print(x_train.shape)
            xgb_classifier = xgb.XGBClassifier()
            xgb_classifier.fit(x_train, y_train)
            y_pred = xgb_classifier.predict(x_test)
            xgb_classifier = accuracy_score(y_test, y_pred)
            msg = 'Accuracy of xgboost:  ' + str(xgb_classifier*100)
            return render(request, 'train.html', {'acc': msg})

        elif model == "3":
            clf = RandomForestClassifier()
            clf.fit(x_train._get_numeric_data(), np.ravel(y_train, order='C'))
            y_pred = clf.predict(x_test)
            clf = accuracy_score(y_test, y_pred)
            msg = 'Accuracy of RandomForest : ' + str(clf*100)
            return render(request, 'train.html', {'acc': msg})

        elif model == "4":
            gnb = GaussianNB()
            gnb.fit(x_train, y_train)
            y_pred = gnb.predict(x_test)
            clf = accuracy_score(y_test, y_pred)
            msg = 'Accuracy of GaussianNB : ' + str(clf*100)
            return render(request, 'train.html', {'acc': msg})

        elif model == "5":
            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            clf = accuracy_score(y_test, y_pred)
            msg = 'Accuracy of LogisticRegression : ' + str(clf*100)
            return render(request, 'train.html', {'acc': msg})

        elif model == "6":
            # model = keras.Sequential()
            # model.add(Dense(30, activation='relu'))
            # model.add(Dense(20, activation='relu'))
            # model.add(Dense(1, activation='softmax'))
            # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # model.fit(x_train, y_train, batch_size=100, epochs=20,validation_data=(x_test, y_test))
            # abc=model.predict(x_test)
            # acc =accuracy_score(abc,y_test)
            #  + str(acc)
            msg = 'Accuracy of ANN :  ' + str()*100
            # print(classification_report(y_test, model.predict))
            return render(request, 'train.html', {'acc': msg})

        elif model == "7":
            svm_classifier = svm()
            svm_classifier.fit(x_train, y_train)
            y_pred = svm_classifier.predict(x_test)
            svm_classifier = accuracy_score(y_test, y_pred)
            msg = 'Accuracy of svm:  ' + str(svm_classifier*100)
            return render(request, 'train.html', {'acc': msg})

        elif model == "8":
            knn_classifier = KNeighborsClassifier()
            knn_classifier.fit(x_train, y_train)
            y_pred = knn_classifier.predict(x_test)
            knn_classifier = accuracy_score(y_test, y_pred)
            msg = 'Accuracy of knn:  ' + str(knn_classifier*100)
            return render(request, 'train.html', {'acc': msg})

        elif model == "9":
            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            clf = accuracy_score(y_test, y_pred)
            msg = 'Accuracy of LSTM : ' + str(clf*100)
            return render(request, 'train.html', {'acc': msg})

        elif model == "10":
            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            clf = accuracy_score(y_test, y_pred)
            msg = 'Accuracy of BILSTM : ' + str(clf*100)
            return render(request, 'train.html', {'acc': msg})

        elif model == "11":
            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            clf = accuracy_score(y_test, y_pred)
            msg = 'Accuracy of GRU : ' + str(clf*100)
            return render(request, 'train.html', {'acc': msg})

        elif model == "12":
            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            clf = accuracy_score(y_test, y_pred)
            msg = 'Accuracy of CNN : ' + str(clf*100)
            return render(request, 'train.html', {'acc': msg})


    return render(request, 'train.html')

# Predicting the results.


def predictions(request):

    global data

    if request.method == 'POST':

        N = int(request.POST['N'])
        P = float(request.POST['P'])
        K = float(request.POST['K'])
        temperature = float(request.POST['temperature'])
        humidity = float(request.POST['humidity'])
        ph = float(request.POST['ph'])
        rainfall = float(request.POST['rainfall'])

        PRED = [[N, P, K, temperature, humidity, ph, rainfall]]

        clf = DecisionTreeClassifier()
        clf.fit(x_train, y_train)
        result = clf.predict(PRED)
        result = result[0]
        print(result)

        msg1 = ('The recommended crop is ', result)
        print(msg1)

        #if result == 0:
          #  msg1 = 'WHEAT'
        #else:
            #msg1 = 'maize'

        return render(request, 'predictions.html', {'msg1': msg1})
    return render(request, 'predictions.html')
