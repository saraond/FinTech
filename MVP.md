Loading all necessary packages


```python
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
```

Defining the API key using a throw-away account in order to retrieve the real time data API data. Instead of using this throw-away account, we could also store this key in a file and load the information from there, however, then the professor/teaching assistant would have to load their own API keys. For simplicity  in this asssignment, we go with the first option.
The data will be retrieved from Alpha Vantage, favored for its broad data coverage, ease of use when integrating into Python and its free API access (although limited for free accounts).


```python
API_KEY = 'FN9MWSGO73DP01GS'
BASE_URL = 'https://www.alphavantage.co/query'
```

Retrieving real time data


```python
def get_real_time_data(symbol):
    function = 'GLOBAL_QUOTE'
    params = {
        'function': function,
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return data

```

Example usage: real-time API data retrieval

Input needed: company symbol


```python
symbol = 'AAPL'
```


```python
data = get_real_time_data(symbol)
print(json.dumps(data, indent=4))
```

    {
        "Global Quote": {
            "01. symbol": "AAPL",
            "02. open": "209.1500",
            "03. high": "211.3800",
            "04. low": "208.6100",
            "05. price": "209.0700",
            "06. volume": "56713868",
            "07. latest trading day": "2024-06-25",
            "08. previous close": "208.1400",
            "09. change": "0.9300",
            "10. change percent": "0.4468%"
        }
    }
    

Portfolio rebalancing


```python
def rebalance_portfolio(current_values, desired_weights):

    # Calculate the total portfolio value
    total_value = sum(current_values.values())
    
    # Calculate current weights
    current_weights = {asset: value / total_value for asset, value in current_values.items()}
    
    # Calculate the amount to buy/sell for each asset
    trades = {}
    for asset in current_values.keys():
        desired_value = desired_weights[asset] * total_value
        current_value = current_values[asset]
        trades[asset] = desired_value - current_value
    
    return trades

```

Example usage: Portfolio rebalancing

Input needed: current values of each asset, as well as their corresponding weight as desired by the investor


```python
current_values = {'AAPL': 1000, 'GOOGL': 1500, 'MSFT': 2000}
desired_weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
```


```python
trades = rebalance_portfolio(current_values, desired_weights)
print(trades)
```

    {'AAPL': 800.0, 'GOOGL': -150.0, 'MSFT': -650.0}
    

- : should sell
+ : should buy

Portfolio Return Calculation


```python
def calculate_portfolio_return(initial_values, final_values):

    # Calculate individual asset returns
    asset_returns = {asset: (final_values[asset] - initial_values[asset]) / initial_values[asset]
                     for asset in initial_values}
    
    # Calculate total portfolio return
    initial_portfolio_value = sum(initial_values.values())
    final_portfolio_value = sum(final_values.values())
    portfolio_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
    
    return portfolio_return, asset_returns

```

Input needed: information on initial values per asset as well as their final values


```python
initial_values = {'AAPL': 1000, 'GOOGL': 1500, 'MSFT': 2000}
final_values = {'AAPL': 1200, 'GOOGL': 1600, 'MSFT': 2100}
```


```python
portfolio_return, asset_returns = calculate_portfolio_return(initial_values, final_values)
print(f"Total Portfolio Return: {portfolio_return:.2%}")
print("Individual Asset Returns:")
for asset, return_value in asset_returns.items():
    print(f"{asset}: {return_value:.2%}")
```

    Total Portfolio Return: 8.89%
    Individual Asset Returns:
    AAPL: 20.00%
    GOOGL: 6.67%
    MSFT: 5.00%
    

Satisfictory total return, with Apple stocks being the best performing from the portfolio

Risk assessment

Input needed: tickers of the stocks in the portfolio as well as the start and end dates for the historical data - forward looking techniques may be used too


```python
tickers = ['AAPL', 'GOOGL', 'MSFT']
start_date = '2022-01-01'
end_date = '2023-01-01'
```

Risk and Average Return Calculation


```python
# Fetch the historical data for the stocks
stock_data = yf.download(tickers, start=start_date, end=end_date)['Close']

# Calculate daily returns
returns = stock_data.pct_change()

# Calculate risk (standard deviation of returns) for each stock
risk = returns.std()

# Calculate average return for each stock
average_return = returns.mean()

# Print the results
print("Risk (Std Dev) for each stock:")
print(risk)
print("\nAverage Return for each stock:")
print(average_return)

# Calculate portfolio return assuming equal weights
portfolio_return = returns.mean(axis=1).mean()

# Calculate portfolio risk assuming equal weights
portfolio_risk = returns.mean(axis=1).std()

print(f'\nTotal Portfolio Return: {portfolio_return * 100:.2f}%')
print(f'Total Portfolio Risk (Std Dev): {portfolio_risk}')

```

    [*********************100%%**********************]  3 of 3 completed

    Risk (Std Dev) for each stock:
    Ticker
    AAPL     0.022471
    GOOGL    0.024396
    MSFT     0.022308
    dtype: float64
    
    Average Return for each stock:
    Ticker
    AAPL    -0.001097
    GOOGL   -0.001689
    MSFT    -0.001085
    dtype: float64
    
    Total Portfolio Return: -0.13%
    Total Portfolio Risk (Std Dev): 0.021665889557546347
    

    
    

Annualized risk, Sharpe ratio, Value at Risk -Same input data as for the previous case


```python
# Calculate annualized risk (standard deviation of returns) for each stock
annualized_risk = returns.std() * np.sqrt(252)

# Calculate average annualized return for each stock
average_annualized_return = returns.mean() * 252

# Print the results
print("Annualized Risk (Std Dev) for each stock:")
print(annualized_risk)
print("\nAverage Annualized Return for each stock:")
print(average_annualized_return)

# Calculate the Sharpe Ratio for each stock
risk_free_rate = 0.02
sharpe_ratio = (average_annualized_return - risk_free_rate) / annualized_risk
print("\nSharpe Ratio for each stock:")
print(sharpe_ratio)

# Calculate portfolio metrics assuming equal weights
weights = np.array([1/len(tickers)] * len(tickers))
portfolio_return = np.sum(returns.mean() * weights) * 252
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
portfolio_sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

print(f'\nTotal Portfolio Annualized Return: {portfolio_return * 100:.2f}%')
print(f'Total Portfolio Annualized Risk (Std Dev): {portfolio_volatility}')
print(f'Total Portfolio Sharpe Ratio: {portfolio_sharpe_ratio}')

# Calculate Value at Risk (VaR) at 95% confidence level for the portfolio
VaR_95 = returns.dot(weights).quantile(0.05)
print(f'\nPortfolio Value at Risk (VaR) at 95% confidence level: {VaR_95 * 100:.2f}%')

```

    Annualized Risk (Std Dev) for each stock:
    Ticker
    AAPL     0.356716
    GOOGL    0.387272
    MSFT     0.354123
    dtype: float64
    
    Average Annualized Return for each stock:
    Ticker
    AAPL    -0.276351
    GOOGL   -0.425538
    MSFT    -0.273503
    dtype: float64
    
    Sharpe Ratio for each stock:
    Ticker
    AAPL    -0.830776
    GOOGL   -1.150451
    MSFT    -0.828819
    dtype: float64
    
    Total Portfolio Annualized Return: -32.51%
    Total Portfolio Annualized Risk (Std Dev): 0.3439353342135532
    Total Portfolio Sharpe Ratio: -1.0034753837774995
    
    Portfolio Value at Risk (VaR) at 95% confidence level: -3.67%
    

Recommendation based on risk assessment


```python
# Define thresholds for recommendation
def recommend_portfolio(portfolio_return, portfolio_risk, portfolio_sharpe_ratio, portfolio_var):
    return_good = portfolio_return > 0.05
    return_bad = portfolio_return < 0.00
    
    risk_good = portfolio_risk < 0.20
    risk_bad = portfolio_risk > 0.30
    
    sharpe_good = portfolio_sharpe_ratio > 1
    sharpe_bad = portfolio_sharpe_ratio < 0
    
    var_good = portfolio_var > -0.02
    var_bad = portfolio_var < -0.05

    # Recommendation logic
    if return_good and risk_good and sharpe_good and var_good:
        return "Recommendation: Stick to this portfolio."
    elif return_bad or risk_bad or sharpe_bad or var_bad:
        return "Recommendation: Consider changing this portfolio."
    else:
        return "Recommendation: Monitor this portfolio closely."

# Print results and recommendation
print("Annualized Risk (Std Dev) for each stock:")
print(annualized_risk)
print("\nAverage Annualized Return for each stock:")
print(average_annualized_return)
print("\nSharpe Ratio for each stock:")
print(sharpe_ratio)

print(f'\nTotal Portfolio Annualized Return: {portfolio_return * 100:.2f}%')
print(f'Total Portfolio Annualized Risk (Std Dev): {portfolio_risk}')
print(f'Total Portfolio Sharpe Ratio: {portfolio_sharpe_ratio}')
print(f'\nPortfolio Value at Risk (VaR) at 95% confidence level: {VaR_95 * 100:.2f}%')

# Get recommendation
recommendation = recommend_portfolio(portfolio_return, portfolio_risk, portfolio_sharpe_ratio, VaR_95)
print("\n" + recommendation)

```

    Annualized Risk (Std Dev) for each stock:
    Ticker
    AAPL     0.356716
    GOOGL    0.387272
    MSFT     0.354123
    dtype: float64
    
    Average Annualized Return for each stock:
    Ticker
    AAPL    -0.276351
    GOOGL   -0.425538
    MSFT    -0.273503
    dtype: float64
    
    Sharpe Ratio for each stock:
    Ticker
    AAPL    -0.830776
    GOOGL   -1.150451
    MSFT    -0.828819
    dtype: float64
    
    Total Portfolio Annualized Return: -32.51%
    Total Portfolio Annualized Risk (Std Dev): 0.021665889557546347
    Total Portfolio Sharpe Ratio: -1.0034753837774995
    
    Portfolio Value at Risk (VaR) at 95% confidence level: -3.67%
    
    Recommendation: Consider changing this portfolio.
    

Trade execution: no available API key so the code will not run as it would in practice. Please review how it would work in theory


```python
# Set up Alpaca API credentials
API_KEY = 'your_alpaca_api_key'
API_SECRET = 'your_alpaca_secret_key'
BASE_URL = 'https://paper-api.alpaca.markets'  #paper trading only

# Initialize the Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Function to place an order
def place_order(symbol, qty, side, order_type='market', time_in_force='gtc'):
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=order_type,
        time_in_force=time_in_force
    )
    return order

# Function to check order status
def check_order_status(order_id):
    order = api.get_order(order_id)
    return order

# Function to get account details
def get_account_details():
    account = api.get_account()
    return account

# Place an order: input needed
order = place_order('AAPL', 10, 'buy')
print(f"Order placed: {order.id}")

# Check the status of the placed order
order_status = check_order_status(order.id)
print(f"Order status: {order_status.status}")

# Get account details
account_details = get_account_details()
print(f"Account cash balance: {account_details.cash}")

```
