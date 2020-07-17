"""
Prediction of COVID-19 in Canada for the month of June using data
from the month of May

Feel free to download the updated data from here
https://www.canada.ca/en/public-health/services/diseases/2019-novel-coronavirus-infection.html
then click on "Current situation" link which will show graphs related to data in Canada.
There is a button for downloading the most recent data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

# Read the data
df = pd.read_csv('covid19.csv')

# Change date column to a datetime format
df['date'] = pd.to_datetime(df['date'],format='%d-%m-%Y')

# select rows containing only provinces and save in a new dataframe
# thus remove the rows containing the country overall numbers
df = df[df.prname != 'Canada']

# Group up the provinces numbers by date to have an overall number for the country
df = df.groupby(df['date']).sum().reset_index()

# Number of confirmed cases per day
confirmed_cases = df['numconf']

# Number of fatalities
fatalities_cases = df['numdeaths']

# Number of recovered cases
recovered_cases = df['numrecover']

#Total number of cases
total_number = df['numtotal']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

# Plot confirmed cases in the country
df['numconf'].plot(ax=ax1, color = 'forestgreen')
ax1.set_title("Confirmed cases of COVID-19 in Canada Jan 31 - June 11", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)

# Plot fatalities cases in the country
df['numdeaths'].plot(ax=ax2, color='teal')
ax2.set_title("Fatalities cases of COVID-19 in Canada Jan 31 - June 11", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)
plt.savefig('conf_and_deaths.png')
plt.show()

# Define SEIR Model' equations as functions

# Susceptibility equation dS/dt
def susceptibility(S, I, beta):
    return -beta * I * S

# Exposed equation dE/dt
def exposed(S, E, I, beta,sigma):
    #sigma  - rate of moving from exposed to infected
    return beta * I *S - sigma * E


# Infection equation dI/dt
def infection(S, I, beta, gamma):
    # gamma - rate of moving from infected to recovered
    return beta * I * S - gamma * I


# Recovered equation dR/dt
def recovered(I, gamma):
    return gamma * I

# Runge Kutta Method of 4th degree order for 4 dimensions
# the four dimensions being Susceptible, Exposed, Infected, Recovered
def runge_kutta4(N, a,b,c,d, sus, exp, inf, rec, beta, gamma, hs):
    a1 = susceptibility(a,c, beta)*hs
    b1 = exposed(a, b, c, beta,sigma)*hs
    c1 = infection(a, c, beta, gamma)*hs
    d1 = recovered(c, gamma)*hs

    ak = a + a1*0.5
    bk = b + b1*0.5
    ck = c + c1*0.5
    dk = d + d1*0.05

    a2 = susceptibility(ak, bk, beta)*hs
    b2 = exposed(ak,bk,ck,beta,sigma) *hs
    c2 = infection(ak, bk, beta, gamma)*hs
    d2 = recovered(bk, gamma)*hs

    ak = a + a2*0.5
    bk = b + b2*0.5
    ck = c + c2*0.5
    dk = d + d2*0.5

    a3 = susceptibility(a, c, beta)*hs
    b3 = exposed(a, b, c, beta,sigma)*hs
    c3 = infection(a, c, beta, gamma)*hs
    d3 = recovered(c, gamma)*hs


    ak = a + a3
    bk = b + b3
    ck = c + c3
    dk = d + d3

    a4 = susceptibility(a, c, beta)*hs
    b4 = exposed(a, b, c, beta,sigma)*hs
    c4 = infection(a, c, beta, gamma)*hs
    d4 = recovered(c, gamma)*hs

    a = a + (a1 + 2*(a2 + a3) + a4)/6
    b = b + (b1 + 2*(b2 + b3) + b4)/6
    c = c + (c1 + 2*(c2 + c3) + c4)/6
    d = d + (d1 + 2*(d2 + d3) + d4)/6
    return a, b, c, d# Define SEIR Model' equations as functions

# Susceptibility equation dS/dt
def susceptibility(S, I, beta):
    return -beta * I * S

# Exposed equation dE/dt
def exposed(S, E, I, beta,sigma):
    #sigma  - rate of moving from exposed to infected
    return beta * I *S - sigma * E


# Infection equation dI/dt
def infection(S, I, beta, gamma):
    # gamma - rate of moving from infected to recovered
    return beta * I * S - gamma * I


# Recovered equation dR/dt
def recovered(I, gamma):
    return gamma * I

# Define the initial conditions

def seir_model(N, b0, beta, gamma, hs):
    """
    N - total number of population
    b0 -
    beta = rate of transitioning from Susceptible to Infected
        or transmision rate
    gamma = rate of transitioning from Infected to Recovered
        or infection rate
    hs = step size of the numerical integration
        - smaller step = more calculation + better accuracy
    """

    #initial condition
    a = float(N-1)/N - b0 #susceptibility rate
    b = float((N-1)/N - b0) #exposed rate
    c = float(1)/N +b0    #infected rate
    d = 0 #recovered rate


    sus, exp, inf, rec= [],[],[], []

    #10 000 time-steps
    for i in range(10000):
        sus.append(a)
        exp.append(b)
        inf.append(c)
        rec.append(d)
        a,b,c,d = runge_kutta4(N, a, b, c, d, susceptibility,
                               exposed, infection, recovered,
                               beta, gamma, hs)

    return sus, exp, inf, rec

# Define the parameters
# total number of cases

# Canada Population number  37.59 million
N = 37.59 * (10**6)
b0 = 0
beta = 0.7
gamma = 0.2
hs = 0.1
sigma = 0.5

susc, expo, infe, reco = seir_model(N, b0, beta, gamma, hs)

# Plot figure

plt.figure(figsize=(8,5))
plt.plot(susc, 'b.', label='susceptible');
plt.plot(expo, 'g.', label='exposed');
plt.plot(infe, 'r.', label='infected');
plt.plot(reco, 'c.', label='recovered/deceased');
plt.title("SEIR model - overview of COVID-19 epidemiology")
plt.xlabel("days")
plt.xlim(0,200)
plt.ylabel("Population Percentage")
plt.legend()
plt.savefig('Infection Model over 200 days.png')
plt.show()


le = preprocessing.LabelEncoder()
df['Day_num'] = le.fit_transform(df.date)

# Define train and test dataset

# train data is May
# Create a dataframe that contains information regarding the month of May
# The month of May data are days 67-97
may_month = df[(df['Day_num']>=67)
               & (df['Day_num']<98)].reset_index()

# test data is June
# Create a dataframe that contains information regarding only the month of June
# The month of June data are days 98-107
june_month = df[(df['Day_num']>=98)
                & (df['Day_num']<108)].reset_index()

# Perform Linear regression to predict the number of cases in June
# Define X is the dates from May
X = [may_month['Day_num']]

# Define y is the number of cases in May
y = [may_month['numconf']]

# Define the model and fit the data
model= LinearRegression().fit(X,y)

# Dates to predict are for June
# Define X_predict to contain the next 30 days
X_predict = [list(range(98,129, 1))]
# Predict the number of confirmed cases for the next 30 days
y_predict = model.predict(X_predict)

# Define x_june and y_june to contain the actual data so far
# June days - current days
x_june= [june_month['Day_num']]

# June values - current values
y_june = [june_month['numconf']]

# Initial starting point for June is end of May
C = max(may_month['numconf'])- min(may_month['numconf'])

# Plot figure of the predicted values for June, versus the actual numbers so far (June 11th today)
plt.figure(figsize=(10,7))
plt.scatter(X_predict,y_predict+C, color = 'blue', label = 'June predicted values')
plt.scatter(x_june,y_june, color= 'forestgreen', label = 'June actual values')
plt.xlabel('Days')
plt.ylabel('Number of cases')
plt.title('Values for June 1- June 30')
plt.legend()
plt.savefig('June Predictions and Actual.png')
plt.show()
