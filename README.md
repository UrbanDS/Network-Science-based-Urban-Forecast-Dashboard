# Network-Science-based-Urban-Forecast-Dashboard
code for ARIC 2022 workshop paper - Network Science-based Urban Forecast Dashboard 

simple_time.py is the baseline of prediction

gcn2layer.py is the graph convolutional network model for predicting the POI visits.

The model take the raw data from safegraph and make prediction. You can change the visits_by_day to define a specific time period.

Train-valid-test dataset split are also realized by segment the visits_by_day.
