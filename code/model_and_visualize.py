###############
# ENVIRONMENT #
###############

# Import libraries
import numpy as np
import pandas as pd
import geopandas as geo
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Path where files are stored
basepath = 'Desktop/Insight Project - HELM DateNight/'

# Import shapes 
shp = geo.read_file(basepath + 'census_spdf.shp')
geocodes = pd.read_csv(basepath + '09-23-18-geocodes.csv')

#############################
# GEOSPATIAL VISUALIZATIONS #
############################# 

# All Southern Ontario
#plt.ylim((43,45.75))
#plt.xlim((-80.5,-75))

# Zoom in on Toronto
#plt.ylim((43.6,43.75))
#plt.xlim((-79.45,-79.3))

# Remove NAs for chloropleth
shp.PpDnsty.fillna(0, inplace=True)
shp['PpDnsty'] = shp['PpDnsty'].apply(np.log1p)

# Population density chloropleth
plt.style.use('fivethirtyeight')
shp.plot(column='PpDnsty')
plt.ylim((43.6,43.75))
plt.xlim((-79.45,-79.3))
plt.tick_params(colors='dimgray')
plt.title('Population density', color='dimgray')

# Lat-longs overlayed on map
plt.style.use('fivethirtyeight')
shp.geometry.plot(facecolor='lightblue', edgecolor='dimgray')
plt.ylim((43.6,43.75))
plt.xlim((-79.45,-79.3))
plt.scatter(geocodes.long, geocodes.lat, c='r', s=10, alpha=0.1)
plt.tick_params(colors='dimgray')

# Subscription probabilities chloropleth

#######
# END #
#######