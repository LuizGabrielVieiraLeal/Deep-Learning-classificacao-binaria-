import numpy as np
import pandas as pd
from keras.models import model_from_json


file = open('breast_cancer_classfier.json', 'r') # Lendo o arquivo json
network_structure = file.read() # Carregando a estrutura da rede
file.close()

classfier = model_from_json(network_structure) # Montando classificador
classfier.load_weights('breast_cancer_classfier_weights.h5') # Inserindo os pesos na rede

new_data = np.array([[15.8, 8.34, 118, 900, 0.1, 0.26, 0.08, 0.134, 0.178, 0.2, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])

prediction = classfier.predict(new_data)
prediction = (prediction > 0.8)

# Comparando resultados com a base de dados
prospective = pd.read_csv('entradas-breast.csv') # Atributos previsores
classifications = pd.read_csv('saidas-breast.csv') # Classificações

classfier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

result = classfier.evaluate(prospective, classifications) # Retornando o valor da loss functions e da precisão