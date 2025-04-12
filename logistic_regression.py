import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Read the dataset and remove all observations that have draws at half time
df = pd.read_csv('EPL2014-2024.csv')
df = df[df['HTR'] != 'D']

# Helper function to determine if a comeback occurred
def determine_comeback(row):
    if row['HTR'] == row['FTR']:
        return 0  
    else:
        return 1  

# Apply the helper function to create the target variable
df['ComebackWin'] = df.apply(determine_comeback, axis=1)

# Create a csv of the new dataset with the target variable
df.to_csv('filtered.csv', index=False)

# Feature engineering
df['GoalDiffHT'] = abs(df['HTHG'] - df['HTAG'])
df['HS_Diff'] = df['HS'] - df['AS']
df['HST_Diff'] = df['HST'] - df['AST']
df['HC_Diff'] = df['HC'] - df['AC']
df['HF_Diff'] = df['HF'] - df['AF']
df['HY_Diff'] = df['HY'] - df['AY']
df['HR_Diff'] = df['HR'] - df['AR']

# Select features for the model
features = [
    'GoalDiffHT', 'HS_Diff', 'HST_Diff',
    'HC_Diff', 'HF_Diff', 'HY_Diff', 'HR_Diff'
]

# Train and test split
X = df[features]
y = df['ComebackWin']
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1, stratify=y)

# Create the model and fit it
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Make predictions and calculate probabilities
comeback_probs = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Print the confusion matrix, classification report, and ROC-AUC score
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, comeback_probs)
print(f"ROC-AUC Score: {auc:.3f}")