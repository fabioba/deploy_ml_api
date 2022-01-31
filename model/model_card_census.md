# Census - LR Model

## Model Details
The census.csv has been downloaded from the Udacity/starter folder on Github.
The first stepe is to clean data and prepare a clean version of the file --> census_clean.csv

## Intended Use
It will be used as input for the forecasting model. The aim of the model is to predict salary category. More in detail, this insight is useful for the marketing team to develop
the AD campaign for the next quarter.

## Training Data
The data has been inputed to a logistic regression model, in order to calculate the salary (>=50K or <50K).

## Evaluation Data
The evaluation has been done on the data sliced input dataframe, on each categorical variable.

## Metrics
This model has been evaluated through the following metrics:
- precision 
- recall
- fbeta