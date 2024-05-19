import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def decompose(df, column):
    decomposition = seasonal_decompose(df[column], model='additive', period=12)
    # Get the components
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    # Visualize the components
    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(df[column], label='Original')
    plt.legend()
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend()
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal')
    plt.legend()
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend()
    plt.suptitle(f'Components for Variable: {column}')
    plt.tight_layout()
    plt.show()


# stationarity check
def stationarity(df, column):
    result = adfuller(df[column])
    # The Dickey-Fuller ADF test is often recommended for shorter series because it tends to perform better with limited data
    print("Variable:", column)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    print("Is the series stationary?", result[1] <= 0.05)


# to adjust p & q in ARIMA
def search_pq(df, column):
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plot_acf(df[column], lags=30, ax=plt.gca())  # ACF
    plt.subplot(212)
    plot_pacf(df[column], lags=30, ax=plt.gca())  # PACF
    plt.suptitle(f'ACF & PACF for Variable: {column}')
    plt.tight_layout()
    plt.show()