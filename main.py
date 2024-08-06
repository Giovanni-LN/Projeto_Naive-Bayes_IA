import numpy as np
import pandas as pd


def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y] == i]) / len(df))
    return prior


def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label, alpha=1):
    df = df[df[Y] == label]
    total_count = len(df)
    feature_count = len(df[df[feat_name] == feat_val])
    # Apply Laplace Smoothing
    p_x_given_y = (feature_count + alpha) / (total_count + alpha * len(df[feat_name].unique()))
    return p_x_given_y


def naive_bayes_categorical(df, X, Y, alpha=1):
    features = list(df.columns)
    features.remove(Y)  # Remove the class column from the features list

    # Calculate prior probabilities for each class
    prior = calculate_prior(df, Y)
    print(f"Prior probabilities: {dict(zip(sorted(df[Y].unique()), prior))}")

    # List of possible class labels
    labels = sorted(list(df[Y].unique()))

    # Store predictions for each sample
    Y_pred = []

    # Loop over each sample in X
    for x in X:
        print(f"Processing sample: {x}")

        # Calculate likelihood for each class
        likelihood = [1] * len(labels)
        for j in range(len(labels)):
            print(f"Class: {labels[j]}")
            for i, feature in enumerate(features):
                feat_val = x[i]
                if pd.isna(feat_val):  # Check for missing values
                    continue
                p_x_given_y = calculate_likelihood_categorical(df, feature, feat_val, Y, labels[j], alpha)
                likelihood[j] *= p_x_given_y
                print(
                    f"Likelihood of feature '{feature}' with value '{feat_val}' for class '{labels[j]}': {p_x_given_y}")

        # Calculate posterior probability (numerator only)
        post_prob = [1] * len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        # Normalize the posterior probabilities
        total = sum(post_prob)
        if total > 0:
            post_prob = [p / total for p in post_prob]
        else:
            post_prob = [0] * len(labels)
        print(f"Likelihood for each class: {dict(zip(labels, likelihood))}")
        print(f"Posterior probabilities: {dict(zip(labels, post_prob))}")

        # Append the class with the highest posterior probability
        Y_pred.append(labels[np.argmax(post_prob)])

    return np.array(Y_pred)


# Example of how to load and preprocess the data
def preprocess_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast',
                  'breast-quad', 'irradiat']
    df.replace('?', np.nan, inplace=True)
    return df


# Example usage
file_path = '/home/giovanni/Documentos/IA/breast-cancer.data'
df = preprocess_data(file_path)

# Define some test data
X = [
    ['40-49', 'premeno', '30-34', '0-2', 'yes', '3', 'left', 'left_low', 'no'],
    # Add more test samples if needed
]

Y = 'Class'
predictions = naive_bayes_categorical(df, X, Y)
print(f"Predictions: {predictions}")