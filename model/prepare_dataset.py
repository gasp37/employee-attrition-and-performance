import pandas as pd
from os import remove

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

df_treated.to_csv('./model/treated_data.csv', index=False)