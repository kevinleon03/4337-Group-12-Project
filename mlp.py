import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=1, stratify=y
)

# Create the model and fit it
mlp_model = MLPClassifier(
    hidden_layer_sizes=(32, 16),   
    activation='relu',            
    solver='adam',               
    max_iter=500,                 
    random_state=1,
    early_stopping=True,           
    validation_fraction=0.1,      
)
mlp_model.fit(X_train, y_train)

# Make predictions
comeback_probs = mlp_model.predict_proba(X_test)[:, 1]
y_pred = mlp_model.predict(X_test)

# Print the confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, comeback_probs)
print(f"ROC-AUC Score: {auc:.3f}")


# Create a grid of hyperparameters to test
learning_rates = [0.001, 0.01, 0.1]
epochs = [100, 300, 500]

results = []

# Loop through the hyperparameters
for lr in learning_rates:
    for epoch in epochs:
        
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            learning_rate_init=lr,
            max_iter=epoch,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=1,
            verbose=False
        )
        
        mlp_model.fit(X_train, y_train)
        
        # Predict
        y_pred = mlp_model.predict(X_test)
        comeback_probs = mlp_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, comeback_probs)

        results.append({
            'learning_rate': lr,
            'epochs': epoch,
            'accuracy': acc,
            'roc_auc': auc,
            'loss_curve': mlp_model.loss_curve_
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Plot Loss Curves
fig, ax = plt.subplots(figsize=(10, 6))

for idx, row in results_df.iterrows():
    loss_curve = row['loss_curve']
    label = f"LR={row['learning_rate']}, Epochs={row['epochs']}"
    ax.plot(loss_curve, label=label)

ax.set_title('Loss Curves for Different Settings')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()