import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display_functions import display

# Set a standard style for the plots
sns.set_style("whitegrid")

# Load the training data from the zip file path
df = pd.read_csv('train.csv')

# Display the first few rows to confirm loading
print("First 5 rows of the dataset:")
display(df.head())

print("--- Data Structure and Missing Values ---")
df.info()

print("\n--- Descriptive Statistics for Numerical Features ---")
print(df.describe())

print("\n--- Value Counts for Key Categorical Features ---")
print("Passenger Class (Pclass):\n", df['Pclass'].value_counts())
print("\nGender (Sex):\n", df['Sex'].value_counts())
print("\nPort of Embarkation (Embarked):\n", df['Embarked'].value_counts())

plt.figure(figsize=(12, 5))

# Age Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['Age'].dropna(), kde=True, bins=30)
plt.title('Distribution of Age')

# Fare Distribution (using log transformation)
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['Fare']), kde=True, bins=30)
plt.title('Distribution of Log(Fare)')

plt.suptitle('Univariate Analysis of Numerical Features')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
plt.show()

# Write observations below this cell (e.g., Age is somewhat normal, Fare is highly skewed)

plt.figure(figsize=(15, 5))

# Survival Rate by Sex
plt.subplot(1, 3, 1)
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')

# Survival Rate by Pclass
plt.subplot(1, 3, 2)
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Pclass')

# Survival Rate by Embarked
plt.subplot(1, 3, 3)
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title('Survival Rate by Port of Embarkation')

plt.suptitle('Bivariate Analysis: Categorical Features vs. Survival')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Write observations below this cell (e.g., Females and 1st class passengers had the highest survival rates)

plt.figure(figsize=(12, 5))

# Age vs. Survival
plt.subplot(1, 2, 1)
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age Distribution by Survival (0=Died, 1=Survived)')

# Fare vs. Survival
plt.subplot(1, 2, 2)
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare Distribution by Survival (0=Died, 1=Survived)')
plt.yscale('log') # Use log scale for clarity due to Fare outliers

plt.suptitle('Bivariate Analysis: Numerical Features vs. Survival')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Select numerical features
numerical_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Write observations below this cell (e.g., Pclass is strongly negatively correlated with Survived)

# Adjust numerical columns to exclude the one-hot encoded Pclass if desired, or keep to show all relationships
sns.pairplot(df[numerical_cols].dropna(), hue='Survived', diag_kind='kde')
plt.suptitle('Pair Plot of Numerical Features Colored by Survival', y=1.02)
plt.show()
