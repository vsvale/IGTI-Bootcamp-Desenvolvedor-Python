import numpy as np
import joblib
from flask import Flask,request,jsonify,render_template

app = Flask(__name__)

def previsao_diabetes(lista_valores_formulario):
    prever = lista_valores_formulario.reshape(1,8)
    modelo_salvo = joblib.load('MLP.sav')
    resultado = modelo_salvo.predict(prever)
    return resultado[0]

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/result',methods=['POST'])
# def