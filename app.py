from flask import Flask, render_template, request

import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Load the pre-trained model
model = keras.models.load_model('covid.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve user input from the form
    continent = int(request.form['continent'])
    country = int(request.form['country'])
    population = float(request.form['population'])
    cases_new = float(request.form['cases_new'])
    cases_active = float(request.form['cases_active'])
    cases_critical = float(request.form['cases_critical'])
    cases_recovered = float(request.form['cases_recovered'])
    cases_1M_pop = float(request.form['cases_1M_pop'])
    cases_total = float(request.form['cases_total'])
    deaths_new = float(request.form['deaths_new'])
    deaths_1M_pop = float(request.form['deaths_1M_pop'])
    tests_1M_pop = float(request.form['tests_1M_pop'])
    tests_total = float(request.form['tests_total'])

    # Prepare input data as a NumPy array
    new_data = np.array([[continent, country, population, cases_new, cases_active, cases_critical, cases_recovered, cases_1M_pop, cases_total, deaths_new, deaths_1M_pop, tests_1M_pop, tests_total]])

    # Make predictions
    new_predictions = model.predict(new_data)

    # Render the result template with the prediction
    return render_template('result.html', prediction=new_predictions[0][0])

if __name__ == '__main__':
    app.run(debug=True)
