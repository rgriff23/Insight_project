###############
# ENVIRONMENT #
###############

# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Path where files are stored
basepath = '~/Desktop/Insight Project - HELM DateNight/'

# Import data
data = pd.read_csv(basepath + '09-27-18_parent_census_data.csv')

#############
# DATA PREP #
#############

# Subset to data for modeling
data = data.drop(['lat',
                        'long',
                        'parent_registration_id',
                        'subscribed',
                        'Area',
                        'Households',
                        'Population'], axis=1)

# Drop rows with infinite/missing values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Log transform variables
data['PopDensity'] = data['PopDensity'].apply(np.log1p)
data['MedianIncome'] = data['MedianIncome'].apply(np.log1p)

########
# VIFs #
########

# VIF for PopDensity
X = data.drop(['PopDensity'], axis=1)
y = data['PopDensity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=37)
model = LinearRegression()
model.fit(X_train, y_train)
R2 = model.score(X_train, y_train)
1/(1-R2) # 1.58

# VIF for CouplesChildren
X = data.drop(['CouplesChildren'], axis=1)
y = data['CouplesChildren']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=37)
model = LinearRegression()
model.fit(X_train, y_train)
R2 = model.score(X_train, y_train)
1/(1-R2) # 2.93

# VIF for MedianIncome
X = data.drop(['MedianIncome'], axis=1)
y = data['MedianIncome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=37)
model = LinearRegression()
model.fit(X_train, y_train)
R2 = model.score(X_train, y_train)
1/(1-R2) # 2.13

# VIF for AveNumRooms
X = data.drop(['AveNumRooms'], axis=1)
y = data['AveNumRooms']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=37)
model = LinearRegression()
model.fit(X_train, y_train)
R2 = model.score(X_train, y_train)
1/(1-R2) # 5.05

#######
# END #
#######