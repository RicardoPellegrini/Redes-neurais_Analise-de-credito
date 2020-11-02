
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Importando database
base = pd.read_csv('credit_data.csv')


# In[3]:


# Estatísticas do database
base.describe()


# In[4]:


# Amostra dos dados
base.head()


# In[5]:


# Verificando dados com idade negativa
base.loc[base['age'] < 0]


# In[6]:


### Maneiras de contornar o problema das idades menores que zero

## 1) Apagar a coluna por inteiro (não recomendada, neste caso)
# base.drop('age', 1, inplace=True)

## 2) Apagar apenas os registros, por completo, que possuem essa incoerência
# base.drop(base[base.age < 0].index, inplace=True)

## 3) Preencher os valores com a média da coluna, apenas dos valores maiores que zero
media = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = media


# In[7]:


# Verificando valores nulos
base.loc[pd.isnull(base['age'])]


# In[8]:


# Divisão do dataset entre variáveis preditoras e target
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values


# In[9]:


# Substituindo os valores missing pela média de cada coluna
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(previsores[:, 0:3])

previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])


# In[10]:


## Fazendo o escalonamento (normalização) dos atributos
from sklearn.preprocessing import StandardScaler

# Padronização
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Normalização
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# previsores = scaler.fit_transform(previsores)


# In[11]:


# Dividindo os dados em treino e teste
from sklearn.model_selection import train_test_split


# In[12]:


previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.25, random_state=0)


# In[13]:


# Modelo Redes Neurais com Keras
import tensorflow


# In[14]:


from tensorflow import keras


# In[15]:


classificador = keras.Sequential()


# In[16]:


# Configurar camadas da nossa rede neural
# Definindo nossa primeira camada oculta, com dois neurônios, recebendo 3 entradas
classificador.add(keras.layers.Dense(units=2, activation='relu', input_dim=3))


# In[17]:


# Definindo nossa segunda camada oculta, com dois neurônios
classificador.add(keras.layers.Dense(units=2, activation='relu'))


# In[18]:


# Definindo nossa camada de saída, com 1 neurônio (problema binário)
classificador.add(keras.layers.Dense(units=1, activation='sigmoid'))


# In[19]:


# Compilando nossa rede neural
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[20]:


# Efetuando o treinamento do modelo, reajustando os pesos a cada 10 registros por 100 vezes
classificador.fit(previsores_train, classe_train, batch_size=10, epochs=100)


# In[21]:


# Testando o modelo criado à partir dos dados de treinamento
previsoes = classificador.predict(previsores_test)
previsoes = (previsoes > 0.5)


# In[22]:


# Calculando a precisão do nosso modelo
from sklearn.metrics import confusion_matrix, accuracy_score


# In[23]:


precisao = accuracy_score(classe_test, previsoes)
precisao


# In[24]:


matriz = confusion_matrix(classe_test, previsoes)
matriz


# ## Resultado
# ### Redes Neurais Tensorflow Keras
# 0.998
