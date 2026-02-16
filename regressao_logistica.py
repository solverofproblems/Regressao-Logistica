#Aqui temos todas as bibliotecas que são necessárias para obter os resultados esperados.
import seaborn as sns 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Aqui estamos utilizando a base de dados dos passageiros do Titanic que já vem junto quando você instala a biblioteca "seaborn".
dados_titanic = sns.load_dataset('titanic')

#Como essa base de dados tem vários valores nulos, nós limpamos ela selecionando as colunas que desejamos e fazemos a limpeza.
dados_limpos = dados_titanic[['survived','pclass','sex','age']].dropna()

#Aqui nós estamos substituindo os textos "male" e "female" para "0" e "1", respectivamente. Isso é importante, pois, a Regressão Logística só trabalha com dados numéricos. 
dados_limpos['sex'] = dados_limpos['sex'].map({'male' : 0, 'female' : 1})

#Aqui estamos definindo as colunas que serão usadas para fazer a previsão:
X = dados_limpos[['pclass', 'sex', 'age']]

#Aqui estamos definindo aquilo que o algoritmo deve prever com base nas colunas especificadas acima. No nosso caso, queremos que ele use a classe, sexo e a idade dos indivíduos para prever se eles sobreviverão, ou não.
Y = dados_limpos[['survived']]

#Aqui estamos definindo os dados de treino e teste. Sobre isso, estamos destinando 20% dos dados para teste. Além disso, estamos definindo uma randomização fixa de 42, evitando altas discrepâncias quando o código é rodado outras vezes.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#Aqui estamos definindo o modelo em si. Usaremos regressão logística aqui.
modelo = LogisticRegression()

#Aqui estamos treinando o modelo com os valores de treino que nós definimos anteriormente.
modelo.fit(X_train, Y_train.values.ravel())

#Aqui estamos exibindo o peso de cada coluna. Quanto maior e positivo o valor for, significa que ele impacta positivamente no percentual final. A mesma lógica se aplica para valores menores e negativos, porém inversa... Valores assim impactam negativamente no percentual final.

#Você verá aqui que o gênero é o fator mais importante, tendo mais de 2 pontos positivos de importância. Entretanto, também verá que a classe que o passageiro se encontra (primeira, segunda ou terceira) tem aproximadamente 1.5 pontos negativos de importância.
print('Peso das variáveis: ')
for var, peso in zip(X.columns, modelo.coef_[0]):
    print(f'{var}:{peso:.4f}')

#Aqui estamos testando o modelo incrementando alguém novo no Titanic com base em algumas informações para saber se ele sobreviveria, ou não.

#Sobre isso, veja que temos um Array da seguinte forma: [[x, y, z]]. 
# X = Classe do indivíduo, podendo ser de 1 à 3. Se for 1 = Primeira Classe (Luxo), 2 = Segunda Classe e 3 = Terceira Classe (econômica).
# Y = Sexo. Se for 1 = Mulher, se for 0 = Homem.
# Z = Idade.
novo_passageiro = pd.DataFrame([[1, 0, 25]], columns=['pclass', 'sex', 'age'])
prob = modelo.predict_proba(novo_passageiro)[0][1]

#Aqui estamos exibindo o percentual daquele indivíduo sobreviver.
print(f'\nProbabilidade de sobrevivência: {prob:.2%}')


#Essa última parte foi incrementada para mostrar como o Threshold atua. Veja que ele é o responsável por "bater o martelo" ao receber uma probabilidade. Eu defini, neste caso, que se a probabilidade de sobrevivência for maior ou igual a 60%, automaticamente o modelo afirma que o indivíduo sobreviveu. Do contrário, afirma que não sobreviveu. 
threshold = 0.6

if prob >= threshold:
    print(f'Sobreviveu. Veja a probabilidade: {prob * 100:.2f}%')
else:
    print(f'Não sobreviveu. Veja a probabilidade: {prob * 100:.2f}%')