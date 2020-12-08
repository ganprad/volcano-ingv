# Repository for doing some time series feature engineering with the INGV Volcano prediction competition data:
https://www.kaggle.com/c/predict-volcanic-eruptions-ingv-oe

The goal of the competition is the build a regression model that can accurately
identify time to an eruption event using time series recordings from 10 sensors.

### Summary:
    + Some of the sensors don't have any data. These are filled with zeros.
    + Overall strategy is to build features using the time series data that will be used by a model.
    + Approach highlighted here is based on dynamic mode decomposition.
    
References: 
- dmdbook.com
- J. N. Kutz, S. L. Brunton, B. W. Brunton, and J. L. Proctor. Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems. SIAM, 2016.
- https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/dynamic-mode-decomposition-of-numerical-and-experimental-data/AA4C763B525515AD4521A6CC5E10DBD4
- http://dmdbook.com/
