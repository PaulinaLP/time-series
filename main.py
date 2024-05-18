from Code import analysis
from Code import arima_exp
from Code import varmax_exp
from Code import create_synthetic_df
import os
import sys
import pandas as pd
import json

script_path = os.path.abspath(os.path.dirname(sys.argv[0]))
output_path = os.path.join(script_path, 'output')
input_path = os.path.join(script_path, 'input')

ANALYSIS = 0
ARIMA_EXP = 0
VARMAX_EXP = 1
SYNTHETIC = 1

if __name__ == '__main__':
    with open(os.path.join(input_path, 'config.json')) as file:
        configuration = json.load(file)
    ext_features = configuration['ext']
    int_features = configuration['int']
    if SYNTHETIC==1:
        df_train=create_synthetic_df.create_df()
    else:
        df_train = pd.read_csv(os.path.join(input_path, 'train.csv'))
    df_without_covid = df_train[((df_train['date'] >= pd.to_datetime('2020-06-30')) | (df_train['date'] <= pd.to_datetime('2020-02-01')))]
    if ANALYSIS == 1:
        for df in ([df_train, df_without_covid]):
            print(df)
            for column in df.columns:
                analysis.decompose(df,column)
                analysis.stationarity(df,column)
                analysis.search_pq(df,column)
    if ARIMA_EXP == 1:
        for column in ext_features:
            experiment_name = "Arima"+str(column)
            uri = configuration['mlflow_uri']
            p_grid = [1, 12]
            q_grid = [1, 12]
            d_grid = [0, 1, 2]
            steps_grid = [6, 12]
            arima_exp.arima_experiment(uri, experiment_name, p_grid, q_grid, d_grid, steps_grid, column, df_train)
    if VARMAX_EXP == 1:
        experiment_name = "VARMAX"
        uri = configuration['mlflow_uri']
        p_grid = [1, 12]
        q_grid = [1, 12]
        steps_grid = [6, 12]
        varmax_exp.varmax_experiment(uri, experiment_name, p_grid, q_grid, steps_grid, int_features, ext_features, df_train)




