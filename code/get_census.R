# Import Dissemination Area (DA) level data

###############
# ENVIRONMENT #
###############

# Load packages
library('cancensus')
library('tidyverse')
library('sf')

# Provide path to a cache folder
options(cancensus.cache_path = '')

# Set API key (https://censusmapper.ca/users/sign_up)
key <- ''
options(cancensus.api_key, key)

###############
# DEFINITIONS #
###############

# Variables to collect
names <- c('Population, 2016',
           'Population percentage change, 2011 to 2016',
           'Population density per square kilometre',
           'Couples with children',
           'Average family size of economic families',
           'Median total income of households in 2015 ($)',
           'Median total income of couple economic families with children in 2015 ($)',
           'Median total income of lone-parent economic families in 2015 ($)',
           'No fixed workplace address',
           'Average number of rooms per dwelling',
           'Average weeks worked in reference year',
           '45 to 59 minutes',
           '60 minutes and over',
           'No certificate, diploma or degree',
           'Secondary (high) school diploma or equivalency certificate',
           "Bachelor's degree",
           "Master's degree",
           'Degree in medicine, dentistry, veterinary medicine or optometry',
           'Earned doctorate'
)
variables <- list_census_vectors('CA16') %>% 
  filter(type == 'Total') # ignore sex segregated data
names %in% unique(variables$label) # check that all the variables are in there
vectors <- variables %>% 
  filter(label %in% names) %>% 
  pull('vector') 
vectors = vectors[-(16:21)]
test = variables[variables$vector %in% vectors,'label']  # check these are what we want
c(test)$label %in% names + names %in% c(test)$label

# Subset to data inside Ontario (PI_UID == 35)
regions <- list_census_regions('CA16') 
regions_list <- regions %>% filter(PR_UID==35)

#########
# QUERY #
#########

# Get census data
census_data <- get_census(dataset='CA16', 
                          regions=regions_list, 
                          vectors=vectors, 
                          level='DA', 
                          geo_format = 'sf',
                          api_key=key)

#############
# VISUALIZE #
#############

# Check that it looks okay
plot(census_data[17], main='Couples with children', ylim=c(42.5,46.5), xlim=c(-82,-74))

######################################
# MATCH USER LOCATIONS W CENSUS DATA #
######################################

# Import user data
geocodes <- read.csv('geocodes.csv')
subscribed <- read.csv('data_subscription_charges.csv')

# Combine geocodes with data on parents (subscribed, booked)
map2(geocodes$long, geocodes$lat, ~st_point(c(.x, .y))) %>% 
  st_sfc(crs = 4326) %>% 
  st_sf(geocodes[,-1], .) -> centers_sf
user_data <- bind_cols(geocodes, census_data[as.numeric(st_within(centers_sf, census_data)),])

##############
# WRITE DATA #
##############

# Write shapefile (for making maps)
geometry <- census_data$geometry
spdf <- as_Spatial(census_data)
writeOGR(spdf, dsn='~/', layer='spdf2', driver='ESRI Shapefile')

# Write matched parents & census data (for modeling)
census_data <- census_data %>% select(-c('geometry'))
write.csv(census_data, 'parent_census_data.csv', row.names=FALSE)

#######
# END #
#######
