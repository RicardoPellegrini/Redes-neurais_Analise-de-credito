
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


# Modelo Redes Neurais
from sklearn.neural_network import MLPClassifier


# In[41]:


classificador = MLPClassifier(
    verbose=True,
    max_iter=500,
    tol=0.00001,
    solver='adam',
    hidden_layer_sizes=(100),
    activation='relu',
    )


# In[42]:


classificador.fit(previsores_train, classe_train)


# In[43]:


# Testando o modelo criado à partir dos dados de treinamento
previsoes = classificador.predict(previsores_test)


# In[44]:


# Calculando a precisão do nosso modelo
from sklearn.metrics import confusion_matrix, accuracy_score


# In[45]:


precisao = accuracy_score(classe_test, previsoes)
precisao


# In[46]:


matriz = confusion_matrix(classe_test, previsoes)
matriz


# ## Resultado
# ### Redes Neurais (MLP Classifier - multi layer perceptron)
# 0.996 - max_iter=200, activation='relu', tol=0.0001, solver='adam'    
# 0.996 - max_iter=500, activation='relu', tol=0.00001, solver='adam'  
# 
