import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Load data
df = pd.read_csv('../data/boston.csv')

# Separate features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Handle missing value
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and preprocessing objects
joblib.dump(model, '../app/model.pkl')
joblib.dump(scaler, '../app/scaler.pkl')
joblib.dump(imputer, '../app/imputer.pkl')
