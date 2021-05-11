import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    toarr=np.array(final_features)
    prediction = model.predict(toarr)
   
    output = (prediction[0])

    return render_template('index.html', prediction_text='Loan approval status is:{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)