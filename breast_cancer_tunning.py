import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


prospective = pd.read_csv('entradas-breast.csv') # Atributos previsores
classifications = pd.read_csv('saidas-breast.csv') # Classificações

def build_neural_network(optimizer, loss, kernel_initializer, activation, neurons):
    classfier = Sequential()

    classfier.add(Dense(
                units=neurons, # Número de neuronios da camada oculta (numEntradas + numSaidas) / 2
                activation=activation, # Função de ativação
                kernel_initializer=kernel_initializer, # Modo de inicialização dos pesos
                input_dim=30 # Parâmetro utilizado somente na primeira camada oculta para especificar o numero de neuronios na camada de entrada
            )) # Adição de camada oculta
    
    # Previnindo underfitting e overfitting
    classfier.add(Dropout(0.2)) # O recomendado é que os atributos sejam entre 20 e 30 porcento zerados
    
    classfier.add(Dense(
                units=neurons, # Número de neuronios da camada oculta (numEntradas + numSaidas) / 2
                activation=activation, # Função de ativação
                kernel_initializer=kernel_initializer, # Modo de inicialização dos pesos
            )) # Adição de camada oculta
    
    # Previnindo underfitting e overfitting
    classfier.add(Dropout(0.2)) # O recomendado é que os atributos sejam entre 20 e 30 porcento zerados
    
    classfier.add(Dense(
                units=1,
                activation='sigmoid'
            )) # Adição da camada de saída
    
    classfier.compile(
                optimizer=optimizer, # Função que será utilizada para fazer o ajuste dos pesos (adam - otimização da descida do gradiente estocastico)
                loss=loss, # Função de perda (tratamento do erro)
                metrics=['binary_accuracy'] # Métricas 
            ) # Compilando a estrutura da rede neural
    
    return classfier

classfier = KerasClassifier(build_fn=build_neural_network)

# Definindo parametrização para testes
params = {'batch_size': [10, 15, 20, 25],
        'epochs': [100, 150, 200], 
        'optimizer': ['adam', 'sgd'], 
        'loss': ['binary_crossentropy', 'hinge'],
        'kernel_initializer': ['random_uniform', 'normal'],
        'activation': ['relu', 'tanh'],
        'neurons': [16, 20]}

grid_search = GridSearchCV(estimator=classfier, param_grid=params, scoring='accuracy', cv=5)

grid_search = grid_search.fit(prospective, classifications)

best_params = grid_search.best_params_ # Melhor parametrização
best_score = grid_search.best_score_ # Melhor resultado