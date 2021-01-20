import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
count_vec = pickle.load(open('count_vec.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sentence = request.form['message']
    word_list = [sentence]

        
    vectorized_features = count_vec.fit_transform(word_list).todense()
    
    prediction = model.predict(vectorized_features)
    
    for i in prediction:
        if i ==0:
            
            out = 'false'
        elif i == 1:
            out = 'barely-true'
        elif i ==2:
            out = 'mostly-true'
        elif i ==3:
            out = 'half-true'
        elif i ==4:
            out = 'pants-fire'
        elif i ==5:
            out = 'true'
            

    output = out

    return render_template('index.html', prediction_text='Probable sentiment of sentence is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)