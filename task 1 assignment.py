import pandas as pd

# Load the dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
try:
    df = pd.read_csv(url)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the URL or file path.")
except Exception as e:
    print(f"An error occurred: {e}")
df.head()
# Data types
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())
# Drop rows with missing values
df_cleaned = df.dropna()

# Alternatively, fill missing values (example: with mean)
# df_filled = df.fillna(df.mean())

df_cleaned.describe()
df_cleaned.groupby('species').mean()

import matplotlib.pyplot as plt
import seaborn as sns

avg_petal_length = df_cleaned.groupby('species')['petal_length'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.lineplot(data=avg_petal_length, x='species', y='petal_length', marker='o')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length')
plt.show()

avg_sepal_width = df_cleaned.groupby('species')['sepal_width'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=avg_sepal_width, x='species', y='sepal_width')
plt.title('Average Sepal Width by Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Width')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df_cleaned['petal_length'], bins=20, kde=True)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_cleaned, x='sepal_length', y='petal_length', hue='species')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(title='Species')
plt.show()


