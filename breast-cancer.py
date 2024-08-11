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

    if feat_val == '?':
        unique_vals = df_label[feat_name].dropna().unique()
        p_x_given_y = 1 / len(unique_vals)
    else:
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
                if feat_val == '?':
                    continue  # Ignora o cálculo se o valor for '?'
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
    df.columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast',
                  'breast-quad', 'irradiat']
    df.replace('?', np.nan, inplace=True)
    return df

def evaluate_accuracy(Y_true, Y_pred):
    """Avalia a acurácia comparando as previsões com as classes reais."""
    correct = np.sum(Y_true == Y_pred)
    accuracy = correct / len(Y_true)
    return accuracy

def prompt_for_more_data():
    """Solicita ao usuário se deseja inserir mais dados e valida a resposta."""
    while True:
        user_input = input("Deseja inserir mais dados? (s/n): ").strip().lower()
        if user_input in ['s', 'n']:
            return user_input
        else:
            print("Resposta inválida. Por favor, insira 's' para sim ou 'n' para não.")

# Carrega e pré-processa os dados
file_path = '/home/giovanni/Documentos/IA/breast-cancer.data'
df = preprocess_data(file_path)

# Divide os dados em conjuntos de treinamento (66%) e teste (34%)
train_df, test_df = train_test_split(df, test_size=0.34, random_state=42)

# Prepara os dados de treinamento
X_train = train_df.drop(columns=['Class']).values
Y_train = train_df['Class'].values

# Prepara os dados de teste
X_test = test_df.drop(columns=['Class']).values
Y_test = test_df['Class'].values

# Treina e testa o classificador Naive Bayes
predictions_train = naive_bayes_categorical(train_df, X_train, 'Class')
predictions_test = naive_bayes_categorical(test_df, X_test, 'Class')

# Calcula a acurácia
accuracy_train = evaluate_accuracy(Y_train, predictions_train)
accuracy_test = evaluate_accuracy(Y_test, predictions_test)

print(f"Acurácia no conjunto de treinamento: {accuracy_train:.2%}")
print(f"Acurácia no conjunto de teste: {accuracy_test:.2%}")

# Exemplo de previsão para uma nova amostra
def predict_new_sample(df, user_input):
    """Faz uma previsão para uma nova amostra fornecida pelo usuário."""
    prediction = naive_bayes_categorical(df, [user_input], 'Class')
    return prediction[0]

# Função de entrada do usuário com validação
def get_user_input():
    """Obtém a entrada do usuário com validação dos valores."""
    valid_values = {
        'age': ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '?'],
        'menopause': ['lt40', 'ge40', 'premeno', '?'],
        'tumor-size': ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '?'],
        'inv-nodes': ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39', '?'],
        'node-caps': ['yes', 'no', '?'],
        'deg-malig': ['1', '2', '3', '?'],
        'breast': ['left', 'right', '?'],
        'breast-quad': ['left_up', 'left_low', 'right_up', 'right_low', 'central', '?'],
        'irradiat': ['yes', 'no', '?']
    }

    print("\nDigite os valores para as características ou '?' para ignorar:")
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

    # Avaliação após a inserção de novos dados
    predictions_train = naive_bayes_categorical(train_df, X_train, 'Class')
    predictions_test = naive_bayes_categorical(test_df, X_test, 'Class')
    accuracy_train = evaluate_accuracy(Y_train, predictions_train)
    accuracy_test = evaluate_accuracy(Y_test, predictions_test)

    print(f"\nAvaliação após atualização do modelo:")
    print(f"Acurácia no conjunto de treinamento: {accuracy_train:.2%}")
    print(f"Acurácia no conjunto de teste: {accuracy_test:.2%}")

    # Solicita se o usuário deseja inserir mais dados
    user_response = prompt_for_more_data()
    if user_response == 'n':
        print("Você optou por não inserir mais dados. Encerrando o programa.")
        break