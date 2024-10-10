
from flask import Flask, render_template, request,make_response
from notebooks import crop_recommendation_model
from utils.fertilizer import fertilizer_dic
from markupsafe import Markup
import numpy as np
import pandas as pd
import requests
import config
import pickle



def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

app = Flask(__name__)

# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# Home page
@app.route('/')
def home():
    title = 'CropSense - Home'
    return render_template('index.html', title=title)

# About page route
@app.route('/about')
def about():
    return render_template('about.html')

# Product page route
crops = [
    'Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas',
    'Mothbeans', 'Mungbean', 'Blackgram', 'Lentil', 'Cotton',
    'Jute', 'Coffee', 'Pomegranate', 'Banana', 'Mango', 'Grapes',
    'Watermelon', 'Muskmelon', 'Apple', 'Orange', 'Papaya', 'Coconut'
]
@app.route('/product')
def product():
    return render_template('product.html', crops=crops)

# Service page route
@app.route('/service')
def service():
    return render_template('service.html')

# Contact page route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        response = make_response(render_template('contact.html'))
        response.set_cookie('cookieName', 'cookieValue', samesite='None', secure=True)
        return response
    else:
        return render_template('contact.html')
    
# Crop Page
@app.route('/crop-recommend')
def crop_recommend():
    title = 'CropSense - Crop Recommendation'
    return render_template('crop.html', title=title)

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'CropSense - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        state = request.form.get("stt")
        city = request.form.get("city")

        weather_data = weather_fetch(city)
        if weather_data is not None:
            temperature, humidity = weather_data
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, pred='images/crop/'+final_prediction+'.jpg')
        else:
            return render_template('try_again.html', title=title)
        
# Fertilizer page        
@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'CropSense - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'CropSense - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

if __name__ == '__main__':
    app.run(debug=True)