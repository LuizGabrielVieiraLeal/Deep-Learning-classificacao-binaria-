import pandas as pd


prospective = pd.read_csv('entradas-breast.csv') # Atributos previsores
classifications = pd.read_csv('saidas-breast.csv') # Classificações


from sklearn.model_selection import train_test_split # Função que faz a divisão da base de dados entre treinamento e teste


prospective_trainment, prospective_test, classifications_trainment, classifications_test = train_test_split(prospective, classifications, test_size=0.25) # test_size indica a porcentagem de registro que será usado para o teste (75% para treinar e 25% para testar)


# Estrutura da rede neural
import keras
from keras.models import Sequential
from keras.layers import Dense # Camada fully connected


classfier = Sequential()

classfier.add(Dense(
            units=16, # Número de neuronios da camada oculta (numEntradas + numSaidas) / 2
            activation='relu', # Função de ativação
            kernel_initializer='random_uniform', # Modo de inicialização dos pesos
            input_dim=30 # Parâmetro utilizado somente na primeira camada oculta para especificar o numero de neuronios na camada de entrada
        )) # Adição de camada oculta

classfier.add(Dense(
            units=16, # Número de neuronios da camada oculta (numEntradas + numSaidas) / 2
            activation='relu', # Função de ativação
            kernel_initializer='random_uniform', # Modo de inicialização dos pesos
        )) # Adição de camada oculta

classfier.add(Dense(
            units=1,
            activation='sigmoid'
        )) # Adição da camada de saída

"""
classfier.compile(
            optimizer='adam', # Função que será utilizada para fazer o ajuste dos pesos (adam - otimização da descida do gradiente estocastico)
            loss='binary_crossentropy', # Função de perda (tratamento do erro)
            metrics=['binary_accuracy'] # Métricas 
        )
"""

# Configurando optimizadores do keras
optimizer = keras.optimizers.Adam(
            lr=0.001, # Taxa de aprendizagem
            decay=0.0001, # Decaimento (decremento) da taxa de aprendizagem
            clipvalue=0.5 # Prende o valor dos pesos e evita que fiquem dispersos no gradiente
        )

classfier.compile(
            optimizer=optimizer, # Função que será utilizada para fazer o ajuste dos pesos (adam - otimização da descida do gradiente estocastico)
            loss='binary_crossentropy', # Função de perda (tratamento do erro)
            metrics=['binary_accuracy'] # Métricas 
        ) # Compilando a estrutura da rede neural

classfier.fit(
            prospective_trainment, 
            classifications_trainment, 
            batch_size=10, # Número de registros por lote para a analise da descida de gradiente
            epochs=100 # Número de épocas de treinamento
        ) # Realizando o treinamento

# Visualização dos pesos
synapses_to_first_hidden_layer = classfier.layers[0].get_weights()
synapses_to_second_hidden_layer = classfier.layers[1].get_weights()
synapses_to_output_layer = classfier.layers[2].get_weights()

# Realizando as previsões na base de dados de teste
forecasts = classfier.predict(prospective_test)
forecasts = (forecasts > 0.5) # Transformando os valores em booleanos para facilitar a comparação


# Medindo a taxa de acerto
from sklearn.metrics import confusion_matrix, accuracy_score


# 1° forma de analisar a taxa de acerto (sklearn) 
precision = accuracy_score(classifications_test, forecasts) # Taxa de acerto
matrix = confusion_matrix(classifications_test, forecasts) # Numero de acertos e erros
# 2° forma de analisar a taxa de acerto (keras) 
result = classfier.evaluate(prospective_test, classifications_test) # 1° linha do retorno -> função de erro, 2° linha de retorno -> precisão