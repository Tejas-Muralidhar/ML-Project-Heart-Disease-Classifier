import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


df = pd.read_csv('heart.csv')


X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

categorical_cols = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols)
    ])

classifiers = {
    'Logistic Regression': LogisticRegression(C=1.0, random_state=65), 
    'Support Vector Machine': SVC(C=1.0, kernel='rbf', random_state=65),  
    'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=65),  
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=2, max_features='sqrt', bootstrap=True, random_state=65)  
}


for name, classifier in classifiers.items():
    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', classifier)])
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    filename = f'{name.lower().replace(" ", "_")}_model.pkl'
    joblib.dump(model, filename)
    
    print(f"Saved {name} model as {filename}")
    
    print('\n####################')
    print(f"\n{name}:\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

