import mlflow
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import math


def arima_experiment(uri, experiment_name, p_grid, q_grid, d_grid, steps_grid, col, df):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment (experiment_name)
    for p in p_grid:
        for q in q_grid:
            for d in d_grid:
                for steps in steps_grid:
                    with mlflow.start_run():
                        mlflow.set_tag("variable", col)
                        mlflow.log_param("p", p)
                        mlflow.log_param("q", q)
                        mlflow.log_param("d", d)
                        mlflow.log_param("steps", steps)
                        largo = df.shape[0]
                        train = df.head(largo - steps)
                        test = df.tail(steps)
                        train[col] = train[col].astype(float)
                        model = ARIMA(train[col], order=(p, d, q))
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=steps)
                        forecast = pd.DataFrame(forecast)
                        joined_df = test.join(forecast)
                        print()
                        print(f'{col} p:{p} q:{q} d:{d}')
                        print(joined_df)
                        min_value = joined_df[col].min()
                        max_value = joined_df[col].max()
                        dif = max_value - min_value
                        mse = ((joined_df[col] - joined_df['predicted_mean']) ** 2).mean()
                        rmse = math.sqrt(mse)
                        error_percent = rmse / dif
                        mlflow.log_param("dif", dif)
                        mlflow.log_metric("rmse", rmse)
                        mlflow.log_metric("error_percent", error_percent)
                        print(f'rmse {rmse}')
                        plt.figure(figsize=(10, 6))
                        plt.plot(joined_df['date'], joined_df[col], label='Actual Values')
                        plt.plot(joined_df['date'], joined_df['predicted_mean'], label='Predicted Values', color='orange')
                        plt.xlabel('Date')
                        plt.ylabel('Value')
                        plt.title(f'Actual vs. Predicted Values {col} p:{p} q:{q} d:{d}')
                        plt.legend()
                        plt.grid()
                        plt.show()
