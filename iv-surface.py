import jax.numpy as jnp
from jax import grad
from jax.scipy.stats import norm as jnorm
from blackscholes import BlackScholesCall
from blackscholes import BlackScholesPut
import blackscholes as bs

# Define Black-Scholes Function
def black_scholes(S, K, T, r, sigma, q=0, otype="call"):
    d1 = (jnp.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    if otype == "call":
        call = S * jnp.exp(-q * T) * jnorm.cdf(d1, 0, 1) - K * jnp.exp(-r * T) * jnorm.cdf(d2, 0, 1)
        return call
    elif otype == "put":
        put = K * jnp.exp(-r * T) * jnorm.cdf(-d2, 0, 1) - S * jnp.exp(-q * T) * jnorm.cdf(-d1, 0, 1)
        return put

# Test Input Code
S = 100.00
K = 110.00
T = 0.8
r = 0.05
sigma = 0.2
q = 0.0

ref_call = bs.BlackScholesCall(S, K, T, r, sigma, q)
ref_call_price = ref_call.price()

ref_put = bs.BlackScholesPut(S, K, T, r, sigma, q)
ref_put_price = ref_put.price()

our_call = black_scholes(S, K, T, r, sigma, q, otype="call")
our_put = black_scholes(S, K, T, r, sigma, q, otype="put")

print(f"Reference Call Price: {round(ref_call_price)}")
print(f"Our Call Price: {round(our_call)}")
print(f"Reference Put Price: {round(ref_put_price)}")
print(f"Our Put Price: {round(our_put)}")