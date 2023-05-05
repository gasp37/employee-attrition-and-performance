import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from os import chdir
from joblib import dump

seed = 42

chdir('./model')
employees_df = pd.read_csv('treated_data.csv')

X = employees_df.drop(columns='Attrition')
y = employees_df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=seed)

smote = SMOTE(random_state=seed)
X_train, y_train = smote.fit_resample(X_train, y_train)


logit = LogisticRegression(C=0.5, max_iter=500, random_state=seed)
trained_model = logit.fit(X_train, y_train)
prediction = trained_model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
print(f'true negatives: {tn}\nfalse positives: {fp}\nfalse negatives: {fn}\ntrue positives: {tp}')

dump(trained_model, 'model.pkl')