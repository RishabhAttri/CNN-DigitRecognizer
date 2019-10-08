from flask import Flask, render_template, url_for,request,jsonify
import pickle
import numpy as np
from scipy.misc import imsave, imread, imresize
import re
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():

    # print(int_features)
    int_features = request.args['Data']
    
    final_features = []
    value = 0
    for x in int_features:
        if x != ',':
            value = value * 10 + int(x)
        else :
            final_features.append(value)
            value = 0
    final_features.append(value)
    for i in range(len(final_features)):
        final_features[i] = final_features[i]/255
        if final_features[i] > 1:
            final_features[i] = 1
    # print(final_features)
    myArray = np.asarray(final_features)
    print(myArray.shape)
    newArray = []
    newArray.append(myArray)
    newArray1 = np.asarray(newArray)
    prediction = model.predict(newArray1)
    print(newArray1)
    print(prediction)
    value = 0
    index = 0
    for i in range(prediction[0].size):
        if value < prediction[0][i]:
            value = prediction[0][i]
            index = i
    return "The Answer is {}".format(index)
if __name__ == '__main__':
    app.run(debug=True)