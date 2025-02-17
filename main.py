import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
import string

# Função de pré-processamento de texto
def preprocess_text(text):
    text = text.lower()  # Convertendo para minúsculas
    text = ''.join([char for char in text if char not in string.punctuation])  # Remover pontuação
    return text

# Carregar dados de spam do arquivo JSON
data = pd.read_json("spam.json")
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Aplicar pré-processamento no texto
data['email'] = data['email'].apply(preprocess_text)

# Criando um pipeline com TfidfVectorizer e Naive Bayes
pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Dividindo em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(data['email'], data['label'], test_size=0.2, random_state=42)

# Treinando o modelo
pipeline.fit(X_train, y_train)

# Testando o modelo
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Exibindo métricas de desempenho
print(f'Acurácia: {accuracy * 100:.2f}%')
print(f'Precisão: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-Score: {f1 * 100:.2f}%')

# Função para testar com novos e-mails
def prever_email(novo_email):
    novo_email = preprocess_text(novo_email)  # Pré-processando o novo e-mail
    resultado = pipeline.predict([novo_email])
    return "É spam" if resultado[0] == 1 else "Não é spam"

# Teste com um e-mail realista
novo_email = input("Digite o email que você quer averiguar: ")
print(f'O e-mail: "{novo_email}" {prever_email(novo_email)}')
