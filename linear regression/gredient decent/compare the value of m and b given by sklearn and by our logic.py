''' Good students always try to solve exercise on their own first and then look at the ready made solution
    I know you are an awesome student !! :)
    Hence you will look into this code only after you have done your due diligence.
    If you are not an awesome student who is full of laziness then only you will come here
    without writing single line of code on your own. In that case anyways you are going to
    face my anger with fire and fury !!!
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

def predict_using_sklean():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()    # create object
    r.fit(df[['math']],df.cs)   # train the model with data
    return r.coef_, r.intercept_   # return the value of m and b

def gradient_descent(x,y): 
    m_curr = 0  # start from 0 the value of m and b
    b_curr = 0
    iterations = 1000000 # yeti steps heu
    n = len(x)  # shows the number of item in each column
    learning_rate = 0.0002  # each steo ko length

    cost_previous = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr  # here the x is the list calling values
        cost = (1/n)*sum([value**2 for value in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):  # purano rw naya cost value 1e-20 value le close chan vaney break the loop
            break
        cost_previous = cost # if not then new value is stored in cost_previous
        print ("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost, i))

    return m_curr, b_curr

if __name__ == "__main__":
    df = pd.read_csv("test_scores.csv")   # defining the dataframe
    x = np.array(df.math)   # select only math column
    y = np.array(df.cs)   # select only cs column

    m, b = gradient_descent(x,y)    # calling the function and passing the math column and cs column
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_using_sklean()  #calling second function
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))
