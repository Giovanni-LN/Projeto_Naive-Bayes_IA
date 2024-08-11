import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def calculate_prior(df, Y):
    """Calcula as probabilidades a priori para cada classe."""
    class_counts = df[Y].value_counts()
    total_count = len(df)
    prior = {cls: count / total_count for cls, count in class_counts.items()}
    return prior

def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label, alpha=1):
    """Calcula a verossimilhança para dados categóricos."""
    df_label = df[df[Y] == label]
    total_count = len(df_label)

    feature_count = len(df_label[df_label[feat_name] == feat_val])
    unique_vals = df_label[feat_name].dropna().unique()
    p_x_given_y = (feature_count + alpha) / (total_count + alpha * len(unique_vals))

    return p_x_given_y

def naive_bayes_categorical(df, X, Y, alpha=1):
    """Implementa o classificador Naive Bayes para dados categóricos."""
    features = list(df.columns)
    features.remove(Y)

    prior = calculate_prior(df, Y)
    labels = sorted(list(df[Y].unique()))

    Y_pred = []

    for x in X:
        likelihood = np.ones(len(labels))
        for j, label in enumerate(labels):
            for i, feature in enumerate(features):
                feat_val = x[i]
                p_x_given_y = calculate_likelihood_categorical(df, feature, feat_val, Y, label, alpha)
                likelihood[j] *= p_x_given_y

        post_prob = likelihood * np.array([prior[label] for label in labels])
        total = post_prob.sum()
        if total > 0:
            post_prob = post_prob / total
        else:
            post_prob = np.zeros(len(labels))

        Y_pred.append(labels[np.argmax(post_prob)])

    return np.array(Y_pred)

def preprocess_data(file_path):
    """Carrega e pré-processa os dados."""
    df = pd.read_csv(file_path, header=None)
    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    return df

def evaluate_accuracy(Y_true, Y_pred):
    """Avalia a acurácia comparando as previsões com as classes reais."""
    correct = np.sum(Y_true == Y_pred)
    accuracy = correct / len(Y_true)
    return accuracy

# Carrega e pré-processa os dados
file_path = '/home/giovanni/Documentos/IA2/car.data'  # Atualize este caminho para o local do seu arquivo
df = preprocess_data(file_path)

# Divide os dados em conjuntos de treinamento (66%) e teste (34%)
train_df, test_df = train_test_split(df, test_size=0.34, random_state=42)

# Prepara os dados de treinamento
X_train = train_df.drop(columns=['class']).values
Y_train = train_df['class'].values

# Prepara os dados de teste
X_test = test_df.drop(columns=['class']).values
Y_test = test_df['class'].values

# Treina e testa o classificador Naive Bayes
predictions_train = naive_bayes_categorical(train_df, X_train, 'class')
predictions_test = naive_bayes_categorical(test_df, X_test, 'class')

# Calcula a acurácia
accuracy_train = evaluate_accuracy(Y_train, predictions_train)
accuracy_test = evaluate_accuracy(Y_test, predictions_test)

print(f"Acurácia no conjunto de treinamento: {accuracy_train:.2%}")
print(f"Acurácia no conjunto de teste: {accuracy_test:.2%}")

# Exemplo de previsão para uma nova amostra
def predict_new_sample(df, user_input):
    """Faz uma previsão para uma nova amostra fornecida pelo usuário."""
    prediction = naive_bayes_categorical(df, [user_input], 'class')
    return prediction[0]

# Função de entrada do usuário com validação
def get_user_input():
    """Obtém a entrada do usuário com validação dos valores."""
    valid_values = {
        'buying': ['vhigh', 'high', 'med', 'low'],
        'maint': ['vhigh', 'high', 'med', 'low'],
        'doors': ['2', '3', '4', '5more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high']
    }

    print("\nDigite os valores para as características:")
    user_input = []
    for feature, options in valid_values.items():
        while True:
            value = input(f"{feature} {options}: ").strip()
            if value in options:
                user_input.append(value)
                break
            else:
                print(f"Valor inválido. Por favor, insira um dos seguintes valores: {options}")

    return user_input

# Loop para solicitar mais dados
while True:
    user_input = get_user_input()
    user_prediction = predict_new_sample(train_df, user_input)
    print(f"Previsão para a entrada do usuário: {user_prediction}")

    # Solicita se o usuário deseja inserir mais dados
    user_response = input("Deseja inserir mais dados? (s/n): ").strip().lower()
    if user_response == 'n':
        print("Você optou por não inserir mais dados. Encerrando o programa.")
        break
