from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(name='Heart Disease')

X = heart_disease.data.features
Y = heart_disease.data.targets

print(heart_disease.variables)

counts = heart_disease.data.targets.value_counts().reset_index()
counts.columns = ['class', 'count']
print(counts)

import seaborn as sns
sns.barplot(x='class', y='count', data=counts)

numerical_indexes = [i for i, t in heart_disease.variables.type.items() if t in ['Integer']]
numerical_indexes = numerical_indexes[:-1] # should rather exclude based on heart_disease.variables.role
print(numerical_indexes)

numerical_features = heart_disease.data.features.iloc[:, numerical_indexes]

import pandas as pd
summary = pd.DataFrame({
    'mean': numerical_features.mean(),
    'stddev': numerical_features.std()
})

print(summary)

import matplotlib.pyplot as plt

for col in numerical_features.columns:
    plt.figure(figsize=(6,4))
    plt.hist(numerical_features[col], bins=30, edgecolor="k", alpha=0.7)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

categorical_indexes = [i for i, t in heart_disease.variables.type.items() if t in ['Categorical']]
print(categorical_indexes)
categorical_features = heart_disease.data.features.iloc[:, categorical_indexes]

print(categorical_features)

for col in categorical_features.columns:
    plt.figure(figsize=(6,4))
    plt.hist(categorical_features[col], bins=30, edgecolor="k", alpha=0.7)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

total_dataset = heart_disease.data.original

missing_summary = pd.DataFrame({
    'missing_count': total_dataset.isna().sum(),
    'missing_pct': 100 * total_dataset.isna().mean()
})

print(missing_summary)

