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
from scipy.stats import shapiro
import pylab
import seaborn as sns
import statsmodels.api as sm
import pylab as py
import pandas as pd
from collections import defaultdict

"""
Consider the annual rates of tax return of different companies for the year 2021. 
The 10 observations are in thousands. Negative numbers mean that the company owns tax. 
−0.6 3.1 25.3 −16.8 −7.1 −6.2 25.2 22.6 26.0 
Use these 10 observations to complete the following.

(a) Construct a Q-Q plot. Does the data seem to be normally distributed? Explain."""

x = np.array([1000 * v for v in [-0.6, 3.1, 25.3, -16.8, -7.1, -6.2, 25.2, 22.6, 26.0]])

def qqplot_1a(x):
    sm.qqplot(x, line='45')
    py.title('QQ plot of observations')
    py.show()

"""
in both cases we see that the data is not normally distributed.
we would expect to see a lot of dots in the middle and a few on the edges. but this is not the case.
"""

"""(b) Carry out the Shapiro-Wilk test of. 
Let the significance level be α = 0.1. How do you
reconcile the test result with the QQ-plot."""


def shapiro_test_1b(x):
    stat, p = shapiro(x)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.1
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

"""Statistics=0.856, p=0.087
Sample does not look Gaussian (reject H0)"""

"""(c) Carry out the Shapiro-Wilk test of. 
Let the significance level be α = 0.1. How do you
reconcile the test result with the QQ-plot."""



def linear_regression_1c(correlation, x_mean, x_var, y_mean, y_var):
    a = correlation * np.sqrt(y_var) / np.sqrt(x_var)
    b = y_mean - a*x_mean
    print(f'a value is {a}')
    print(f'b value is {b}')


""""Use three different visual EDA (exploratory data analysis) methods that show how the skull size has changed, if at all, over time. 
Write a short paragraph that refers to all three EDA products and concludes them with respect to the skull size change hypothesis."""

def EDA(data_name_csv):
    # read date from url and convert to dataframe
    df = pd.read_csv(data_name_csv, sep='\t', header=None, names=["MB", "BH", "BL", "NH", "Year"])
    by_year, by_feature, means = get_data()
    #plt.title('Pairplot of the data for each year')
    sns.pairplot(df,hue= 'Year',  kind="scatter", height=1.5, palette=sns.color_palette("tab10", 5), plot_kws={"s":10})
    plt.show()
    # sns.pairplot(df,hue= 'Year',  kind="ref", height=1.5, palette=sns.color_palette("tab10", 5))
    # plt.show()

    MB =[]
    BH = []
    Year = []
    BL = []
    NH =[]

    for key in means.keys():
        MB.append(means[key]['MB'])
        BH.append(means[key]['BH'])
        BL.append(means[key]['BL'])
        NH.append(means[key]['NH'])
        Year.append(key)

    fig, axes = plt.subplots(ncols=4)
    axes[0].plot(Year, MB)
    axes[0].title.set_text('MB')
    axes[0].set_xlabel('Years')
    axes[1].plot(Year, BH)
    axes[1].title.set_text('BH')
    axes[1].set_xlabel('Years')
    axes[2].plot(Year, BL)
    axes[2].title.set_text('BL')
    axes[2].set_xlabel('Years')
    axes[3].plot(Year, NH)
    axes[3].title.set_text('NH')
    axes[3].set_xlabel('Years')
    plt.show()

    box_plot_for_each_feature()

def get_data():
    # read date from url and convert to dataframe
    headers = ["MB", "BH", "BL", "NH", "Year"]
    df = pd.read_csv("data.csv", sep='\t', header=None, names=headers)
    dd = defaultdict(list)
    for row_values in df.values:
        dd[row_values[-1]].append(row_values[:-1])

    dd2 = {}
    for h in headers[:-1]:
        dd2[h] = defaultdict(list)
    for row_values in df.values:
        for h in headers[:-1]:
            dd2[h][row_values[-1]].append(row_values[headers.index(h)])

    by_year = {key: pd.DataFrame(value, dtype="float", columns=headers[:-1]) for key, value in dd.items()}
    by_feature = {key: pd.DataFrame(value) for key, value in dd2.items()}
    means = {key: values.mean(axis=0) for key, values in by_year.items()}
    return by_year ,by_feature, means


def box_plot_for_each_feature():
    # box plot for each feature
    by_year, by_feature, means = get_data()
    fig, axes = plt.subplots(ncols=4)
    for i, (key, values) in enumerate(by_feature.items()):
        axes[i].boxplot(values)
        axes[i].set_xticklabels(list(by_year.keys()))
        axes[i].title.set_text(key)
        axes[i].set_xlabel('years')
    plt.show()

if __name__ == "__main__":
    # qqplot_1a(x)
    # shapiro_test_1b(x)
    # linear_regression_1c(-0.8012, 51, 474.207, 423.333, 6678.16)
    EDA("data.csv")

