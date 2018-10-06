# Replace special characters in the addresses for this to run smoothly

################
# PREPARATIONS #
################

import googlemaps
import numpy as np
import pandas as pd
import seaborn as sns

# API key (https://console.developers.google.com/apis/)
gmaps = googlemaps.Client(key='')

# Path where files are stored
basepath = ''

# Data: registrations (subset to M, L, K)
data_parent_reg = pd.read_csv(basepath + 'initial_parent_registration.csv', encoding='latin1')
data_parent_reg['date_entered'] = pd.to_datetime(data_parent_reg['date_entered'])
data_parent_reg.set_index('date_entered', inplace=True)
data_parent_reg = data_parent_reg[['parent_registration_id','postal_code','city']]
data_parent_reg['postal1'] = data_parent_reg.postal_code.str[0]
data_parent_reg = data_parent_reg.loc[data_parent_reg['postal1'].isin(['M','L','K'])]

# Data: addresses
addresses = pd.read_csv(basepath + 'new_street_file.txt', sep='\t', 
                        names=['parent_registration_id','num','st'])
addresses['st'] = addresses['st'].map(lambda x: x.strip())
addresses = pd.merge(data_parent_reg, addresses, on='parent_registration_id')
addresses['address'] = addresses['num'].astype(str) + ' ' + addresses['st'] + ', ' + addresses['city'] + ', Ontario, ' + addresses['postal_code']
addresses.drop(['city','postal1','num','st'], axis=1, inplace=True)

###########
# GEOCODE #
###########

# Takes results from gmaps.geocode and returns (lat, long)
def get_latlong(geo):
    lat = geo[0]['geometry']['location']['lat']
    long = geo[0]['geometry']['location']['lng'] 
    return((lat, long))
    
# Test
#address = '157 Forman Ave, Ontario, M4S2R9'
#test = gmaps.geocode(address)
#lat, long = get_latlong(test)

# Loop through addresses and get geocodes
Lat = []
Long = []
Address = []
for i in np.arange(len(Lat), addresses.shape[0]):
    result = gmaps.geocode(addresses.address[i])
    if (len(result) > 0):
        lat, long = get_latlong(result)
    else: lat, long = (None, None) 
    Lat.append(lat)
    Long.append(long)
    Address.append(addresses.address[i])
    print(i)

# Combine into new DataFrame
geocodes = pd.DataFrame({'address': Address, 'lat': Lat, 'long': Long, 'parent_registration_id':addresses.parent_registration_id})

# Visual check w lat/long scatterplot
sns.scatterplot(x='long',y='lat',data=geocodes, alpha=0.2)

# Write file
geocodes.to_csv(basepath + 'geocodes.csv', index=False)
          
#######
# END #
#######