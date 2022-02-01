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
- precision: `0.7403127124405167`
- recall: `0.6940726577437859`
- fbeta: `0.7164473684210526`

## Ethical Considerations
The input data contains sensitive information.
Each person that partecipate to this interview must fulfill all those information.
One of the variable that could be updated is the gender, since it accepts only two values: `Male` or `Female`. In my view, this constraints does not allow all people to express their gender identity. 

## Caveats and Recommendations
One of the most relevant suggestions is to include the most values for each categorical variable.
Moreover, each variable should not be empty.