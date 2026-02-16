import seaborn as sns 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dados_titanic = sns.load_dataset('titanic')

dados_limpos = dados_titanic[['survived','pclass','sex','age']].dropna()

dados_limpos['sex'] = dados_limpos['sex'].map({'male' : 0, 'female' : 1})

X = dados_limpos[['pclass', 'sex', 'age']]
Y = dados_limpos[['survived']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

modelo = LogisticRegression()
modelo.fit(X_train, Y_train.values.ravel())

print('Peso das variáveis: ')
for var, peso in zip(X.columns, modelo.coef_[0]):
    print(f'{var}:{peso:.4f}')

novo_passageiro = pd.DataFrame([[1, 0, 25]], columns=['pclass', 'sex', 'age'])
prob = modelo.predict_proba(novo_passageiro)[0][1]

print(f'\nProbabilidade de sobrevivência: {prob:.2%}')

threshold = 0.6

if prob >= threshold:
    print(f'Sobreviveu. Veja a probabilidade: {prob * 100:.2f}%')
else:
    print(f'Não sobreviveu. Veja a probabilidade: {prob * 100:.2f}%')