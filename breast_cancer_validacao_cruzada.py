import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


prospective = pd.read_csv('entradas-breast.csv') # Atributos previsores
classifications = pd.read_csv('saidas-breast.csv') # Classificações

def build_neural_network():
    classfier = Sequential()

    classfier.add(Dense(
                units=16, # Número de neuronios da camada oculta (numEntradas + numSaidas) / 2
                activation='relu', # Função de ativação
                kernel_initializer='random_uniform', # Modo de inicialização dos pesos
                input_dim=30 # Parâmetro utilizado somente na primeira camada oculta para especificar o numero de neuronios na camada de entrada
            )) # Adição de camada oculta
    
    # Previnindo underfitting e overfitting
    classfier.add(Dropout(0.2)) # O recomendado é que os atributos sejam entre 20 e 30 porcento zerados
    
    classfier.add(Dense(
                units=16, # Número de neuronios da camada oculta (numEntradas + numSaidas) / 2
                activation='relu', # Função de ativação
                kernel_initializer='random_uniform', # Modo de inicialização dos pesos
            )) # Adição de camada oculta
    
    # Previnindo underfitting e overfitting
    classfier.add(Dropout(0.2)) # O recomendado é que os atributos sejam entre 20 e 30 porcento zerados
    
    classfier.add(Dense(
                units=1,
                activation='sigmoid'
            )) # Adição da camada de saída
    
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
    
    return classfier

classfier = KerasClassifier(
            build_fn=build_neural_network, # Setando a função de montagem da rede neural
            epochs=100, # Definindo épocas
            batch_size=10 # Particionando os dados em 10 lotes
        )

results = cross_val_score(
            estimator=classfier, # Setando classificador
            X=prospective, # Incica quais são os atributos previsores
            y=classifications, # Setando as classificações para a analise da rede
            cv=10, # Vezes em que serão feitos o teste, em cada teste 1 base será separada para teste e as outras para treinamento, elas alternarão entre sí a cada interação
            scoring='accuracy' # Setando como os resultados devem ser retornados
        )

average = results.mean() # Média de acerto da rede
standard_deviation = results.std() # Desvio padrão dos resultados