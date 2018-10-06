#!/usr/bin/env python

################
# Preparations #
################

# Utilities
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import pickle
import os

# Plotting
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go

# Data pre-processing
import numpy as np
import pandas as pd
import googlemaps
import geopandas as geo
from shapely.geometry import Point
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define app
app = dash.Dash(__name__)
application = app.server

################
# Import stuff #
################

# Google maps API key
google_key = os.environ.get('SECRET_KEY')
gmaps = googlemaps.Client(key=google_key)

# Path where files are stored
basepath = './'

# Unpickle model
pickle_path = basepath + 'logistic_pickle.pkl'
with open(pickle_path, 'rb') as unpickle_model:
    unpickle_model = pickle.load(unpickle_model)

# Import test set probabilities and ground truth 
y_test_prob = pd.read_csv(basepath + 'y_test_prob.csv')

# Import (lat, long) of past users
lat_long = pd.read_csv(basepath + 'lat_long.csv')

# Import shapes 
shp = geo.read_file(basepath + 'census_spdf.shp')

#####################
# Utility functions #
#####################

# Accepts a point and geopandas dataframe
# Returns the index of the point, or NA if it isn't found
def geo_search(point, geometry):
    x = False
    i = 0
    until = len(geometry)
    while x == False:
        if i < until:
            x = point.within(geometry[i])
            if x == False: i += 1
        else: 
            i = float('NaN')
            x = True
    return(i)

# Takes results from gmaps.geocode and returns (lat, long)
def get_latlong(geo):
    lat = geo[0]['geometry']['location']['lat']
    long = geo[0]['geometry']['location']['lng'] 
    return((lat, long))

# Geocode address and find index
def geocode(address):
    result = gmaps.geocode(address)
    if (len(result) > 0):
        lat, long = get_latlong(result)
        point = Point(long, lat)
        result = geo_search(point, shp.geometry)
    else: result = 'invalid address'
    return result

###########
# Figures #
###########

# Distplot
distplot_data = [y_test_prob.y_prob[y_test_prob.y_test==0],
                 y_test_prob.y_prob[y_test_prob.y_test==1]]
distplot_fig = ff.create_distplot(distplot_data, ['Not subscribed','Subscribed'],
                                  bin_size=0.05,
                                  show_rug=False)

##########
# Layout #
##########

# Layout
app.layout = html.Div([

     html.Div([
       html.H1('Selecting users for A/B testing'),
       html.H3('Use slider to change range for inclusion in sample:'),
        dcc.RangeSlider(id='my-slider',
                   min=0,
                   max=0.45,
                   step=0.01,
                   value=[0,0.1],
                   updatemode='drag',
                   pushable=0.01,
                   allowCross=False),
        html.Div(id='slider-output-container'),
       
      html.Div([
        html.H3('Enter a valid Ontario address:'),
        dcc.Input(id='my-address', value='Toronto', type='text'),
        html.Button(id='submit-button', type='submit', children='Submit'),
        html.Div(id='address-output-container'),
        ]),
      
        dcc.Graph(
        id='example-graph',
        figure=distplot_fig),
       dcc.Markdown("Go to [GitHub](https://github.com/rgriff23/Insight_project)")
       
       ])
   ], style = {'width': '500px',
               'padding-left': '80px'})

#############
# Callbacks #
#############

# Address
@app.callback(
    Output(component_id='address-output-container', component_property='children'),
    [Input('submit-button','n_clicks')],
    [State('my-address','value')]
)
def update_output(n_clicks, address):
  result = geocode(address)
  if np.isnan(result):
    result = 'Invalid address'
  else:
    model_input = shp.iloc[result][0:-1]
    model_input[0] = np.log1p(model_input[0])
    model_input[2] = np.log1p(model_input[2])
    model_input = np.array(model_input).reshape(1, -1)
    prob = unpickle_model.predict_proba(model_input)[0][1]
    prob = round(prob, 3)
    result = 'Probability of subscribing: {} '.format(prob)
  return result

# Slider for probability threshold
@app.callback(
    Output(component_id='slider-output-container', component_property='children'),
    [Input(component_id='my-slider', component_property='value')]
)
def update_output(value):
    low = value[0]
    high = value[1]
    above = y_test_prob.y_test[(y_test_prob.y_prob > low) & (y_test_prob.y_prob < high)]
    expected_discounts = round(len(above)/y_test_prob.shape[0] * 100)
    expected_proportion = round(above.sum()/len(above) * 100, 1)
    x = 'You selected a range from {} to {}.'.format(low, high)
    y = 'Per 100 users, ~{} will be included in the sample. The baseline subscription probability of this group is {}%.'.format(expected_discounts, expected_proportion)
    return x + ' ' + y

#######
# End #
#######

if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=5000)
#    app.run_server(debug=True)
