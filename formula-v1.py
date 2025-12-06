import math
from scipy.stats import norm

# Define Variables
S = 50 # Underlying Price of Stock
K = 45 # Strike Price
T = 0.5 # Time to Expiration (Half a Year)
r = 0.1 # Risk-Free Rate
vol = 0.2 # Volatility (σ)

# Calculate d1 and d2
d1 = (math.log(S/K) + (r + 0.5 * vol**2)*T) / (vol * math.sqrt(T))
d2 = d1 - (vol * math.sqrt(T))

# Calculate Call and Put Option Price
C = (S * norm.cdf(d1)) - (K * math.exp(-r * T) * norm.cdf(d2))
P = (K * math.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))

# Print Results
print(f"The price of the call option is ${round(C, 2)}")
print(f"The price of the put option is ${round(P, 2)}")