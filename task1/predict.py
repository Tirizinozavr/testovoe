from joblib import load
import pandas as pd 
import os 

def predict_my_file():
    estimator = load('task1\model.joblib')
    df = pd.read_csv('test_task1.csv')
    results = estimator.predict(df['text_employer'])
    df['ACTION_ITEM_RESULT_PRODUCT_NAME'] = results
    df.to_csv('result.csv')

if __name__=='__main__':
    predict_my_file()