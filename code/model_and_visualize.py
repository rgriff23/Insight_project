###############
# ENVIRONMENT #
###############

# Import libraries
import numpy as np
import pandas as pd
import geopandas as geo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
#import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Path where files are stored
basepath = ''

# Import data for model, clean up missing/inf values
# Log1p transform PopDensity & MedianIncome
data = pd.read_csv(basepath + 'parent_census_data.csv')
data.replace([np.inf, -np.inf], 0, inplace=True)
data.fillna(0, inplace=True)
data = data.drop(['lat','long','parent_registration_id','Area','Households','Population'], axis=1)
data['PopDensity'] = data['PopDensity'].apply(np.log1p)
data['MedianIncome'] = data['MedianIncome'].apply(np.log1p)

# Import shapefiles, clean up missing/inf values
shp = geo.read_file(basepath + 'census_spdf.shp')
shp.replace([np.inf, -np.inf], 0, inplace=True)
shp.fillna(0, inplace=True)

# Import geocoded addresses
geocodes = pd.read_csv(basepath + 'geocodes.csv')

##########################################
# CREATE PRE-PROCESSING PIPELINE & MODEL #
##########################################

# Create pipeline 
scale = StandardScaler()
poly = PolynomialFeatures(interaction_only=True, include_bias = False)
model = LogisticRegression()
pipeline = Pipeline([('scale', scale),
                     ('poly', poly),
                     ('model', model)])

# Create train/test sets 
X = data.drop(['subscribed'], axis=1)
y = data['subscribed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=37)

# Fit the pipeline/model
pipeline.fit(X_train, y_train)
pipeline.predict_proba(X_train)

##################
# EVALUATE MODEL #
##################

# Run RFECV
#rfecv = RFECV(model)
#rfecv = rfecv.fit(X_train, y_train)
#rfecv.n_features_
#rfecv.support_
#rfecv.ranking_
#model = rfecv.estimator_ 

# Fit new reduced model
#X_train = X_train[:,rfecv.support_]
#X_test = X_test[:,rfecv.support_]
#model.fit(X_train, y_train)

# Cross validation score (ROC)
cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc').mean() # 0.65

# Confusion matrix 
predictions = pipeline.predict(X_train)
conf1 = confusion_matrix(y_train, predictions)
(conf1[0,0]+conf1[1,1])/conf1.sum() # 0.75
conf1

# Look at coefficients
model.coef_.round(3)[0]

# Accuracy on test set
predictions_test = pipeline.predict(X_test)
probabilities_test = pipeline.predict_proba(X_test)
confusion_matrix(y_test, predictions_test)
conf_test = confusion_matrix(y_test, predictions_test)
(conf_test[0,0]+conf_test[1,1])/conf_test.sum() # 0.77

###################
# PICKLE PIPELINE #
###################

#pickle_path = 'logistic_pickle.pkl'

# Model
#pickle_model = open(pickle_path, 'wb')
#pickle.dump(pipeline, pickle_model)
#pickle_model.close()

# Write file with predictions + ground truth for app
#probs = pipeline.predict_proba(X_test)[:,1]
#y_test_prob = pd.DataFrame({'y_test': y_test,
#                            'y_prob': probs})
#y_test_prob.to_csv('y_test_prob.csv', index=False)

# Import pickled pipeline
#pickle_path = 'logistic_pickle.pkl'
#with open(pickle_path, 'rb') as pipeline:
#    pipeline = pickle.load(pipeline)

#####################################
# APPLY MODEL TO ALL CENSUS REGIONS #
#####################################

# Loop through regions and compute predicted probability
region_probs = []
for i in np.arange(len(shp)):
    model_input = shp.iloc[i][0:-1]
    model_input = np.array(model_input.astype(float)).reshape(1, -1)
    model_input[0][0] = np.log1p(model_input[0][0])
    model_input[0][2] = np.log1p(model_input[0][2])
    prob = pipeline.predict_proba(model_input)[0][1]
    region_probs = np.append(region_probs, prob)

# Add region probabilities to shp
shp['region_probs'] = region_probs
    
#############################
# NONSPATIAL VISUALIZATIONS #
#############################

# ROC curve
roc_plot = plt.subplot(111)
fpr, tpr, threshold = roc_curve(y_test, probabilities_test[:,1])
roc_auc = auc(fpr, tpr)
plt.style.use('fivethirtyeight')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.title('ROC curve')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# Distribution of estimated probabilities for known subscribers/non-subscribers
plot = pd.DataFrame(y_test)
plot = plot.reset_index()
plot = plot.assign(probs = pd.Series(probabilities_test[:,1]))
sns.distplot(plot.probs[plot.subscribed==1], color='blue')
sns.distplot(plot.probs[plot.subscribed==0], color='red')
plt.style.use('fivethirtyeight')
plt.xlabel('Probabilities')
plt.title('Logistic regression probabilities')
plt.legend(['Subscribed', 'Not subscribed'])

# Barchart comparing true proportions of subscribing in the test data
# and predicted probabilities binned into quartiles
barplot = pd.DataFrame({'y_test':y_test.reset_index().subscribed,
                        'proba':probabilities_test[:,1]})
barplot['Quartile'] = pd.qcut(barplot['proba'], 4, labels=False)
barplot['Quartile'] = barplot['Quartile']+1
plt.style.use('fivethirtyeight')
sns.barplot(x='Quartile', y='y_test', data=barplot)
plt.ylabel('True proportion')
plt.xlabel('Predicted probability quartiles')
plt.hlines(y_test.mean(), linestyles='dashed', xmin=-1, xmax=5)

#############################
# GEOSPATIAL VISUALIZATIONS #
############################# 

# All Southern Ontario
#plt.ylim((43,45.75))
#plt.xlim((-80.5,-75))

# Zoom in on Toronto
#plt.ylim((43.6,43.75))
#plt.xlim((-79.45,-79.3))

# Slightly bigger zoom in on Toronto
#plt.ylim((43.6,43.85))
#plt.xlim((-79.7,-79.2))

# Zoom in on Ottawa
#plt.ylim((45.2,45.5))
#plt.xlim((-75.9,-75.55))

# Remove NAs for chloropleth
shp.PpDnsty.fillna(0, inplace=True)
shp['PpDnsty'] = shp['PpDnsty'].apply(np.log1p)

# Population density chloropleth
plt.style.use('fivethirtyeight')
shp.plot(column='PpDnsty', legend=True)
#plt.ylim((43,45.75))
#plt.xlim((-80.5,-75))
plt.ylim((43.6,43.75))
plt.xlim((-79.45,-79.3))
plt.tick_params(colors='dimgray')
plt.title('Population density', color='dimgray')

# Lat-longs overlayed on map
plt.style.use('fivethirtyeight')
shp.geometry.plot(facecolor='lightblue', edgecolor='dimgray')
#plt.ylim((43,45.75))
#plt.xlim((-80.5,-75))
plt.ylim((43.6,43.75))
plt.xlim((-79.45,-79.3))
plt.scatter(geocodes.long, geocodes.lat, c='r', s=10, alpha=0.1)
plt.tick_params(colors='dimgray')

# Subscription probabilities chloropleth
plt.style.use('fivethirtyeight')
shp.plot(column='region_probs', legend=True)
plt.ylim((43,45.75))
plt.xlim((-80.5,-75))
plt.tick_params(colors='dimgray')
plt.title('Predicted probabilities', color='dimgray')

#######
# END #
#######