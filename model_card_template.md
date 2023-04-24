# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Deployed by Jie who works at Google, 2023, v1
- Logistic Regression Model
- Trained for predict peole's salary based on their census data

## Intended Use
- Intended to be used for complete udacity projects
- Intended to be used for get instance predictions from Heroku

## Training Data
- Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

- Prediction task is to determine whether a person makes over 50K a year.

## Evaluation Data

## Metrics
- Precision, recall and fbeta are used to measure the performance of the model
- precision: 0.685, recall: 0.263, fbeta: 0.381

## Ethical Considerations
- census data is published on UCI's website. No sensitive (name, SSN, etc) data is disclosed

## Caveats and Recommendations
- Does not use complex models (CNN etc)
