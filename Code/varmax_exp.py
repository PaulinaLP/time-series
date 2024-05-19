import mlflow
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error


def varmax_experiment(uri, experiment_name, p_grid, q_grid, steps_grid, endo_col, exo_col, df):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment (experiment_name)
    for p in p_grid:
        for q in q_grid:
            for steps in steps_grid:
                with mlflow.start_run():
                    mlflow.log_param("p", p)
                    mlflow.log_param("q", q)
                    mlflow.log_param("steps", steps)
                    largo = df.shape[0]
                    train = df.head(largo - steps)
                    test = df.tail(steps)
                    model = VARMAX(train[endo_col], exog=train[exo_col], order=(p,q), initialization='stationary' )
                    results = model.fit()
                    # Evaluate model on test data
                    forecast = results.forecast(steps=len(test), exog=test[exo_col])
                    # Compute RMSE for each endogenous column
                    rmse_scores = {}
                    for col in endo_col:
                        rmse = np.sqrt(mean_squared_error(test[col], forecast[col]))
                        rmse_scores[col + '_RMSE'] = rmse
                    # Log RMSE metrics
                    mlflow.log_metrics(rmse_scores)
                    mlflow.log_metrics({
                        "AIC": results.aic,
                        "BIC": results.bic
                    })

