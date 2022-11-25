!pip install latexify-py==0.2.0

import latexify
import math  # Optionally

@latexify.function
def sinc(x):
  if x == 0:
    return 1
  else:
    return math.sin(x) / x

sinc

"""Consider the annual rates of tax return of different companies for the year 2021. 
The 10 observations are in thousands. Negative numbers mean that the company owns tax. 
−0.6 3.1 25.3 −16.8 −7.1 −6.2 25.2 22.6 26.0 
Use these 10 observations to complete the following.

(a) Construct a Q-Q plot. Does the data seem to be normally distributed? Explain."""


"""(b) Carry out the Shapiro-Wilk test of. 
Let the significance level be α = 0.1. How do you
reconcile the test result with the QQ-plot."""