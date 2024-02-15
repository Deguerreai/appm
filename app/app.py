from flask import Flask, request, jsonify,render_template
import pickle
import pandas as pd
import re  # Pour les expressions régulières

app = Flask(__name__)

def extract_number(mixed_input):
    """
    Extraire le nombre d'une entrée mixte. Si l'entrée est déjà un nombre, elle est retournée telle quelle.
    Si l'entrée est une chaîne contenant un nombre, le premier nombre trouvé est extrait et retourné.
    """
    if isinstance(mixed_input, (int, float)):
        return mixed_input
    elif isinstance(mixed_input, str):
        # Recherche de nombres dans la chaîne
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", mixed_input)
        if numbers:
            # Conversion du premier nombre trouvé en float ou int
            number = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
            return number
    return None 

def load_catboost_model(filename='modele_catboost.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction(): 
    data = request.json
    bedrooms = data.get('bedrooms')
    bathrooms = data.get('bathrooms')
    sqft_living = data.get('sqft_living')
    sqft_lot = data.get('sqft_lot')
    floors = data.get('floors')
    waterfront = data.get('waterfront')
    view = data.get('view')
    condition = data.get('condition')
    grade = data.get('grade')
    sqft_above = data.get('sqft_above')
    sqft_basement = data.get('sqft_basement')
    yr_built = data.get('yr_built')
    yr_renovated = data.get('yr_renovated')
    zipcode = data.get('zipcode')
    lat = data.get('lat')
    long = data.get('long')
    sqft_living15 = data.get('sqft_living15')
    sqft_lot15 = data.get('sqft_lot15')
 

    # Traitement des données pour extraire les nombres si nécessaire
    for key, value in data.items():
        data[key] = extract_number(value)

    data_df = pd.DataFrame([data])  # Conversion en DataFrame après traitement
    regression_model = load_catboost_model()  # Chargement du modèle
    prediction = regression_model.predict(data_df)  # Faire la prédiction
    prediction = prediction[0].tolist()
    return jsonify({"prediction": prediction})  # Retourner la prédiction en JSON

if __name__ == '__main__':
    app.run(debug=True)