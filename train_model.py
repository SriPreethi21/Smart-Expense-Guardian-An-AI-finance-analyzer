import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv('data/expenses.csv')

# -------------------------------
# ✅ FULL DATA CLEANING (IMPORTANT)
# -------------------------------

# Remove rows where Amount is missing
df = df.dropna(subset=['Amount'])

# Clean Description
df['Description'] = df['Description'].fillna('unknown')
df['Description'] = df['Description'].astype(str).str.lower().str.strip()

# Clean Category
df['Category'] = df['Category'].fillna('unknown')
df['Category'] = df['Category'].astype(str).str.lower().str.strip()

# Remove empty strings or spaces
df = df[df['Category'] != '']
df = df[df['Category'] != ' ']

# Final safety: remove any remaining NaN
df = df.dropna(subset=['Description', 'Category'])

# Reset index
df = df.reset_index(drop=True)

# -------------------------------
# ✅ ANOMALY MODEL
# -------------------------------
iso = IsolationForest(contamination=0.2, random_state=42)
iso.fit(df[['Amount']])

pickle.dump(iso, open('model/anomaly_model.pkl', 'wb'))

# -------------------------------
# ✅ CATEGORY MODEL
# -------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Description'])
y = df['Category']

model = MultinomialNB()
model.fit(X, y)

pickle.dump(model, open('model/category_model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

print("Models trained successfully ✅")