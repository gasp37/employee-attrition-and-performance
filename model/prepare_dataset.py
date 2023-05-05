import pandas as pd

df_employees = pd.read_csv('./csv-files/WA_Fn-UseC_-HR-Employee-Attrition.csv')

relevante_features = [
    'Attrition',
    'OverTime',
    'TotalWorkingYears',
    'MonthlyIncome',
    'JobInvolvement',
    'JobSatisfaction',
    'EnvironmentSatisfaction'
]

df_treated = df_employees[relevante_features]

boolean_values = {'Yes':1, 'No':0}
df_treated = df_treated.replace(boolean_values)

df_treated.to_csv('./model/treated_data.csv', index=False)