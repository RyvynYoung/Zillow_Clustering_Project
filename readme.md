# Zillow Clustering Project

### Author: Ryvyn Young

## Description: 
- Zillow: What is driving the errors in the Zestimates?
- The presentation will consist of a notebook demo of the discoveries you made and work you have done related to uncovering what the drivers of the error in the zestimate is.

## Instructions for Replication
Files are located in Git Repo [here](https://github.com/RyvynYoung/Zillow_Clustering_Project)
User will need env.py file with access to Codeup database 

## Domain Research:
- What is a single unit housing unit?
    - https://www.investopedia.com/terms/h/housingunits.asp
- What is fips?
    - https://en.wikipedia.org/wiki/FIPS_county_code
- What is the min/max tax rate by county in US?
    - https://www.attomdata.com/news/market-trends/figuresfriday/top-10-u-s-counties-with-the-greatest-effective-tax-rates/
- Understanding Zillow Zestimate
    - https://www.zillow.com/zestimate/
    - https://www.youtube.com/watch?v=rfWzMI_VwTQ
    - https://www.kaggle.com/c/zillow-prize-1/overview


## Key Findings

**Zestimate Error Drivers**
- Confirmed driver = county based on statistical testing and recursive modeling
- Some additional expected drivers based on statistical testing: 
    - tax dollar count, structure dollar per sqft, bedroom count, finished sqft, tax rate, 
    - but these do not necessarily score as key features when using RFE to evaluate

**Best Performing Model**
- The best model was a Linear Regression using all features (excludes cluster names)
    - result is 3.6% worse than baseline

**Possible Next Steps**
- obtain additional data on school rating and distance from beach
- reassess clusters and adjust
- reduce focus to one county
- additional feature engineering and model types


## Project Organization
```
Zillow Culstering Project [repo](XXXXXXXX)
├── README.md               <- The top-level README for developers using this project.
│
├── acquire.py              <- The script to download or generate data
├── prepare.py              <- The script for preparing the raw data
├── summarize.py            <- The script for summarizing the raw data
├── wrangle_zillow.py       <- The script for running the acquire and prepare functions
│                              then splitting and scaling the data
├── explore.py              <- The script to produce visualizations for the notebook
│                              then splitting and scaling the data
├── model.py                <- The script to produce models and return results to the notebook
│
├── mvp.ipynb               <- The draft notebook for the project
│
├── Final.ipynb             <- The finished notebook for presentation of the project
```

## Data Dictionary
see data_dictionary.md

*****
## Planning Stage
Project Description: 
- Use cluster model to identify single unit property groups within the Codeup Zillow 2017 dataset
- Use these clusters to engineer new features for exploration and modeling
- Build, fit, and train a regression model to predict the logerror
- Determine which features most impact the logerror and generate model that results in a smaller mean logerror

GOALS:
- Produce a model that has less logerror than baseline
- Determine key drivers of error

MVP Questions to answer:
- Determine the baseline logerror
- Produce a model that has less logerror than baseline
- Determine key drivers of error

## Acquire Stage
DELIVERABLES: 
- Data is collected from the Codeup cloud database with an appropriate SQL query
- Define single unit property

## Preparation Stage
DELIVERABLES:
- Column data types are appropriate for the data they contain
- Missing values are investigated and handled
- Outliers are investigated and handled

## Exploration and Pre-Processing Stage
DELIVERABLES: 
- Interaction between independent variables and the target variable is explored using visualization and statistical testing
- Clustering is used to explore the data
- A conclusion, supported by statistical testing and visualization, is drawn on whether or not the clusters are helpful/useful
- At least 3 combinations of features for clustering should be tried

## Modeling Stage
DELIVERABLES: 
- At least 4 different models are created and their performance is compared
- One model is the distinct combination of algorithm, hyperparameters, and features
- Best practices on data splitting are followed

## Delivery Stage
DELIVERABLES:
- [Main_Notebook](https://github.com/RyvynYoung/Zillow_Clustering_Project/blob/main/Final.ipynb)
- walk through of notebook 5min
