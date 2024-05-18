import pandas as pd
import numpy as np


def create_df():
    # Define the date range
    dates = pd.date_range(start='2016-01-01', end='2023-12-31', freq='M')

    # Generate synthetic data for each variable
    np.random.seed(0)  # for reproducibility

    euribor = np.random.uniform(low=0.0, high=2.0, size=len(dates)) + 0.5
    unemployment_rate = np.random.uniform(low=10.0, high=25.0, size=len(dates))
    gasoline_price = np.random.uniform(low=1.0, high=2.5, size=len(dates))
    cars_sold_spain = np.random.randint(10000, 50000, size=len(dates))
    cars_sold_company = np.random.randint(2000, 10000, size=len(dates))
    marketing_spent = np.random.randint(100000, 500000, size=len(dates))

    # Create DataFrame
    data = {
        'date': dates,
        'EURIBOR': euribor.astype(float),
        'UnemploymentRate': unemployment_rate.astype(float),
        'GasolinePrice': gasoline_price.astype(float),
        'CarsSoldSpain': cars_sold_spain.astype(float),
        'CarsSoldCompany': cars_sold_company.astype(float),
        'MarketingSpent': marketing_spent.astype(float)
    }

    df = pd.DataFrame(data)

    # Set Date as index
    df.set_index('date', inplace=True, drop=False)

    print(df.head())
    return df
