import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load data
df = pd.read_csv("insurance.csv")

# encoding
df['sex'] = df['sex'].map({'male':0, 'female':1})
df['smoker'] = df['smoker'].map({'no':0, 'yes':1})

# ONE HOT ENCODING (IMPORTANT FIX)
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# features
X = df.drop('charges', axis=1)
y = df['charges']

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LinearRegression()
model.fit(X_train, y_train)

# accuracy
print("Accuracy:", model.score(X_test, y_test))