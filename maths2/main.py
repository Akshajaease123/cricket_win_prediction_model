import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('tableConvert.com_58ya4j.csv')
features = ['Runs', 'Wick', 'OppRuns', 'OppWick']
target = 'Result'

X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()


model.fit(X_train_scaled, y_train)


predictions = model.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions,output_dict=True)


print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)



team_runs = int(input("Enter your team's runs: "))
team_wickets = int(input("Enter your team's wickets: "))
opponent_runs = int(input("Enter opponent's runs: "))
opponent_wickets = int(input("Enter opponent's wickets: "))


next_match_data = pd.DataFrame({
    'Runs': [team_runs],
    'Wick': [team_wickets],
    'OppRuns': [opponent_runs],
    'OppWick': [opponent_wickets]
})

next_match_data_scaled = scaler.transform(next_match_data)


next_match_prediction = model.predict(next_match_data_scaled)

if next_match_prediction[0] == 1:
    print("Your team is predicted to win the match!")
else:
    print("Your team is predicted to lose the match.")