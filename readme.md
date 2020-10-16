# Zillow Clustering Project

### Author: Ryvyn Young

## Description: 
- Zillow: What is driving the errors in the Zestimates?
- The presentation will consist of a notebook demo of the discoveries you made and work you have done related to uncovering what the drivers of the error in the zestimate is.

## Instructions for Replication
Files are located in Git Repo [here](xxxx)
User will need env.py file with access to Codeup database 

## Domain Research:

**What is a single unit property?**     
By James Chen Updated Sep 11, 2020 What Is a Housing Unit?

The term housing unit refers to a single unit within a larger structure that can be used by an individual or household to eat, sleep, and live. The unit can be in any type of residence such as a house, apartment, mobile home, or may also be a single unit in a group of rooms. Essentially, a housing unit is deemed to be a separate living quarter where the occupants live and eat separately from other residents of the structure or building. They also have direct access from the building's exterior or through a common hallway.
- https://www.investopedia.com/terms/h/housingunits.asp

**What is 'fips'?**
This code is use to identify US counties by assigning a number to each county. To look up the county corresponding to the fips number in the dataset add the leading 0 and ignore the decimal. A chart of fips codes can be found [here](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697)
- https://en.wikipedia.org/wiki/FIPS_county_code


## Key Findings

1. **Top 5 drivers of logerror (RMSE)**
- xxxxxx

2. **Top Model = xxxxxx**
- 
  

###### Next Steps    



## Project Organization
```
Zillow Culstering Project [repo](XXXXXXXX)
├── README.md               <- The top-level README for developers using this project.
│
├── acquire.py              <- The script to download or generate data
├── prepare.py              <- The script for preparing the raw data
├── wrangle_zillow.py       <- The script for running the acquire and prepare functions
│                              then splitting and scaling the data
├── explore.py              <- The script to produce visualizations for the notebook
│                              then splitting and scaling the data
├── model.py                <- The script to produce models and return results to the notebook
│
├── mvp.ipynb               <- The draft notebook for the project
│
├── final.ipynb             <- The finished notebook for presentation of the project
```

## Data Dictionary
  ---                            ---
| **Feature**                  | **Definition**                                         |
| ---                          | ---                                                    |
| bathroomcnt                  | # of bathrooms in home                                 |
| bedroomcnt                   | # of bedrooms in home                                  |
| calculatedfinishedsquarefeet | calculated square footage                              |
| fips                         | Code to identify US counties                           |
| latitude                     | Angular distance north or south of the earth's equator |
| longitude                    | Geographic coordinate, east–west position on the Earth |
| lotsizesquarefeet            | lot size in square feet                                |
| taxamount 	                 | Amount paid in taxes in dollars                        |
| yearbuilt                    | year of construction                                   |
| regionidzip                  | zip code                                               |
| propertyuseid                | property land use code                                 |

 
  ---                            ---                                                    
| **Target**                   | **Definition**                                         |
| ---                          | ---                                                    |
| logerror                     | Zestimate error                                        |
***

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
- [Main_Notebook](xxxxxxx)
- walk through of notebook
