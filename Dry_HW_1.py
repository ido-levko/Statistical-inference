# !pip install latexify-py==0.2.0

import latexify
import math  # Optionally

@latexify.function
def sinc(x):
  if x == 0:
    return 1
  else:
    return math.sin(x) / x

sinc

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pylab
import seaborn as sns
import statsmodels.api as sm
import pylab as py

"""
Consider the annual rates of tax return of different companies for the year 2021. 
The 10 observations are in thousands. Negative numbers mean that the company owns tax. 
−0.6 3.1 25.3 −16.8 −7.1 −6.2 25.2 22.6 26.0 
Use these 10 observations to complete the following.

(a) Construct a Q-Q plot. Does the data seem to be normally distributed? Explain."""

x = np.array([1000 * v for v in [-0.6, 3.1, 25.3, -16.8, -7.1, -6.2, 25.2, 22.6, 26.0]])

def qqplot_1a(x):
    sm.qqplot(x, line='45')
    py.show()

"""
in both cases we see that the data is not normally distributed.
we would expect to see a lot of dots in the middle and a few on the edges. but this is not the case.
"""

"""(b) Carry out the Shapiro-Wilk test of. 
Let the significance level be α = 0.1. How do you
reconcile the test result with the QQ-plot."""
from scipy.stats import shapiro

def shapiro_test_1b(x):
    stat, p = shapiro(x)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.1
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')



if __name__ == "__main__":
    # qqplot_1a(x)
    shapiro_test_1b(x)