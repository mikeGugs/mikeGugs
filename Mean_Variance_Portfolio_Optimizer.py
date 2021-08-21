from tiingo import TiingoClient
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from datetime import date, timedelta

# You will need a Tiingo API key, which you can get for free on their website. Insert your API key below in between the quotes where it says 'INSERT-API-KEY-HERE'
client = TiingoClient({'api_key':'INSERT-API-KEY-HERE'})

# Create an empty list to be populated by user input
list_of_stocks = []

# Find out how many stocks the user wants to include
number_of_stocks = int(input('How many stocks would you like to include?'))

# Prompt user to input list of tickers, to appended to list_of_stocks above
print("Enter each ticker you'd like to include, one at a time (pressing <enter> after each ticker), below: ")

# Loop through n iterations defined by user to create list of stocks
for i in range(0,number_of_stocks):
    ele = input()
    list_of_stocks.append(ele)

# Define variables that hold today's date, and the day five years ago to be used in the dataframe of historical prices
today = date.today()
yesterday = today - timedelta(days=1)
yesterday1 = yesterday.strftime('%Y%m%d')
five_years_ago = today - timedelta(days=(365*5))
five_years_ago1 = five_years_ago.strftime('%Y%m%d')

# Create dataframe with user-defined list of stocks of the adjusted closes for 5 years
all_history = client.get_dataframe(list_of_stocks,
                                      metric_name='adjClose',
                                      startDate=five_years_ago1,
                                      endDate=yesterday1,
                                      frequency='daily')

# Calculate daily returns from dataframe
stocks = list(all_history.columns[0:])
daily_returns = all_history[stocks].pct_change()

# find expected return of each stock and annualize them. Express as a percent
expected_return = daily_returns.mean()
annual_exp_return = expected_return * 252
annual_exp_return_pct = annual_exp_return * 100

# find std dev of returns, express as a percent.
std = daily_returns.std()
std_year = std * 252**.5
std_year_pct = std_year * 100

# find covariance matrix of returns
returns_cov = daily_returns.cov()

# find correlation between stocks
returns_corr = daily_returns.corr()

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(all_history)
S = risk_models.sample_cov(all_history)

# Find portfolio optimized for maximal Sharpe ratio
print('These are the weights of the long-only, maximum Sharpe ratio portfolio:')
ef = EfficientFrontier(mu, S, weight_bounds=(0,1))
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.csv")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)
ef = EfficientFrontier(mu, S, weight_bounds=(0,1))

# Create a figure
plt.figure(figsize=(10,5))

# Plot the efficient frontier line
ax3 = plt.subplot(1,2,2)
plotting.plot_efficient_frontier(ef, ax=ax3, show_assets=True)

# find the portfolio optimized for minimum volatility
print('These are the weights of the long-only, minimum volatility portfolio:')
min_vol = ef.min_volatility()
print(min_vol)
performance = ef.portfolio_performance(verbose=True)

# Create side-by-side bar graph comparing volatility (annualized std dev.) vs avg annualized returns for each stock
n=1
t=2
d= number_of_stocks
w=.8
x_values_one = [t * element + w * n for element in range(d)]

n=2
t=2
d= number_of_stocks
w=.8
x_values_two = [t * element + w * n for element in range(d)]

ax = plt.subplot(1,2,1)
plt.bar(x_values_one, annual_exp_return_pct)
plt.bar(x_values_two, std_year_pct)
plt.title('Expected Return vs Volatility')
plt.xlabel('Stocks')
plt.ylabel('Expected Return (annualized) vs Volatility (annualized std dev')
plt.legend(['Expected Return', 'Volatility'])
ax.set_xticks(x_values_one)
ax.set_xticklabels(all_history[stocks])
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# create a graph showing the efficient frontier with 5000 random portfolios
# Find the tangency portfolio
ef.max_sharpe()
ret_tangent, std_tangent, _ = ef.portfolio_performance()
ax2=plt.subplot(1,2,2)
plt.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios and plot them
n_samples = 5000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt(np.diag(w @ S @ w.T))
sharpes = rets / stds
ax1=plt.subplot(1,2,2)
plt.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
ax1.set_title("Efficient Frontier with Random Portfolio Weights")
ax1.legend()
ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1))
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))
plt.tight_layout()

# Print the previous day's closing prices
print("These are the previous day's closing prices:")
latest_prices = get_latest_prices(all_history)
print(latest_prices)

# Find out if user wants to include a $ value of their portfolio to find exact amounts of stock to buy
yes_or_no = input('Do you want to give a dollar value of your portfolio to find the exact amounts of stock to buy? (Answer yes or no)')

# Include portfolio value to come up with exact amount of stocks to buy
if yes_or_no == 'yes':
    dollar_value = int(input('What is the dollar value of your portfolio? (Leave out the dollar sign)'))
    discrete = DiscreteAllocation(cleaned_weights,
                                latest_prices,
                                total_portfolio_value=dollar_value,
                                short_ratio=None)
    allocation, leftover = discrete.greedy_portfolio()
    print('You should buy the following amounts of shares: ', allocation)
    print('This is the amount of money you will have leftover: ${:.2f}'.format(leftover))


# Show the figure
plt.show()
