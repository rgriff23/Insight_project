# Insight Data Science (Fall 2018 session)
## Consulting project for a babysitting app

This repo contains the scripts used for my Insight project. Note that for privacy reasons, data on user locations (e.g., addresses, lat/long coordinates) and API keys are not included in this repo. 

### Summary of repo contents

1. `geocode.py` - geocode user addresses using the [Googlemaps API](https://developers.google.com/maps/documentation/geocoding/start).
2. `get_census_data.R` - request Ontario census data using the [cancensus API](https://cran.r-project.org/web/packages/cancensus/index.html), combining with user data (geocodes, subscription records), and writing shapefiles (for making maps) and user-census data (for modeling).
3. `model_and_visualize.py` - fit logistic regression model and create visualizations. 
x. `VIFs.py` - compute variance inflation factors.


Finally, the 'application' Folder contains files for the dash app (www.datenightdash.com).


