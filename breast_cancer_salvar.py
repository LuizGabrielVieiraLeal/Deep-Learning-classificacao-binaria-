import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout


prospective = pd.read_csv('entradas-breast.csv') # Atributos previsores
classifications = pd.read_csv('saidas-breast.csv') # Classificações

classfier = Sequential()

classfier.add(Dense(
            units=8, 
            activation='relu', 
            kernel_initializer='normal',
            input_dim=30 
        )) 

classfier.add(Dropout(0.2)) 

classfier.add(Dense(
            units=8, 
            activation='relu', 
            kernel_initializer='normal', 
        )) 

classfier.add(Dropout(0.2)) 

classfier.add(Dense(
            units=1,
            activation='sigmoid'
        )) 

classfier.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['binary_accuracy'] 
        ) 

classfier.fit(prospective, classifications, batch_size=10, epochs=100)

classfier_json = classfier.to_json() # Tranformando o classificador em json para salva-lo e não ter que treina-lo toda vez para classificar um registro

with open('breast_cancer_classfier.json', 'w') as file: # Salvando o cassificador
    file.write(classfier_json)
    
classfier.save_weights('breast_cancer_classfier_weights.h5') # Salvando os pesos