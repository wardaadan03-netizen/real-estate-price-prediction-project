import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
data_path = "data/Melbourne_housing_FULL.csv"
df = pd.read_csv(data_path)

# Drop rows with missing Price
df = df.dropna(subset=['Price'])

# Fill missing numeric columns with median
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
numeric_cols.remove('Price')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing categorical columns with 'Unknown'
cat_cols = df.select_dtypes(include='object').columns.tolist()
df[cat_cols] = df[cat_cols].fillna('Unknown')

# Define features and target
target = 'Price'
features = numeric_cols + cat_cols
X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Column transformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Save pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/pipeline.pkl")
print("✅ Pipeline trained and saved to models/pipeline.pkl")