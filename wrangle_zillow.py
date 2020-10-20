import pandas as pd
import numpy as np
import scipy as sp 
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import acquire
import prepare
import summarize
import sklearn


######## Add features ######
def create_features(df):
    df['age'] = 2017 - df.yearbuilt

    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt
    
    # create acres variable
    # df['acres'] = df.lotsizesquarefeet/43560
    
    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet

    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
    
    # ratio of beds to baths
    df['bed_bath_ratio'] = df.bedroomcnt/df.bathroomcnt
    
    # 12447 is the ID for city of LA. 
    # I confirmed through sampling and plotting, as well as looking up a few addresses.
    # df['cola'] = df['regionidcity'].apply(lambda x: 1 if x == 12447.0 else 0)
    
    return df

def remove_outliers(df):
    '''
    remove outliers in tax rate and calculated finished sqft
    '''
    return df[((df.taxrate > .01) & (df.taxrate < .066) & (df.calculatedfinishedsquarefeet < 7000) & (df.lotsizesquarefeet < 2000000))]

####### Split dataframe ########
def split(df, target_var):
    '''
    This splits the dataframe for train, validate, and test, and creates X and y dataframes for each
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state = 123, stratify=df.fips)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state = 123, stratify=train_validate.fips)
    
    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


######## Scale #########

def add_scaled_columns(X_train, X_validate, X_test, scaler, columns_to_scale):
    """This function takes the inputs from scale_zillow and scales the data"""
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(X_train[columns_to_scale])

    X_train_scaled = pd.concat([
        X_train,
        pd.DataFrame(scaler.transform(X_train[columns_to_scale]), columns=new_column_names, index=X_train.index),
    ], axis=1)
    X_validate_scaled = pd.concat([
        X_validate,
        pd.DataFrame(scaler.transform(X_validate[columns_to_scale]), columns=new_column_names, index=X_validate.index),
    ], axis=1)
    X_test_scaled = pd.concat([
        X_test,
        pd.DataFrame(scaler.transform(X_test[columns_to_scale]), columns=new_column_names, index=X_test.index),
    ], axis=1)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled

def scale_zillow(X_train, X_validate, X_test):
    """This function provides the inputs and runs the add_scaled_columns function"""
    X_train_scaled, X_validate_scaled, X_test_scaled = add_scaled_columns(
    X_train,
    X_validate,
    X_test,
    scaler = sklearn.preprocessing.MinMaxScaler(),
    columns_to_scale=['bedroomcnt', 'calculatedfinishedsquarefeet', 'fullbathcnt', 'lotsizesquarefeet', 'roomcnt', 'unitcnt', 
    'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount', 'longitude', 'latitude', 'age', 'taxrate', 'structure_dollar_per_sqft', 
    'land_dollar_per_sqft', 'bed_bath_ratio'],
    )
    return X_train_scaled, X_validate_scaled, X_test_scaled

############# Clustering Exercises ###########

def wrangle_zillow_cluster():
    '''
    Get zillow data and prepare data, return df ready for split and scale with all nulls managed
    '''
    # get data with acquire file
    df = acquire.get_zillow_cluster_data()
    
    # drop known duplicate columns and those with proportion of null values above threshold with prepare file
    df = prepare.data_prep(df, cols_to_remove=['id', 'id.1', 'pid', 'tdate'], prop_required_column=.5, prop_required_row=.5)
    
    # add column that is county name based on fips id
    def get_county_name(county):
        if county == 6037:
            return 'LA'
        elif county == 6059:
            return 'Orange'
        elif county == 6111:
            return 'Ventura'

    df['county'] = df.fips.apply(get_county_name)

    # removing these columns
    #df = prepare.get_counties(df)

    # drop additional columns with nulls that are duplicate or have too many remaining nulls to use
    cols_to_remove2 = ['heatingorsystemtypeid', 'buildingqualitytypeid', 'finishedsquarefeet12', 'propertyzoningdesc', 'regionidcity', 
                        'censustractandblock', 'heatingorsystemdesc', 'calculatedbathnbr', 'propertylandusetypeid', 'assessmentyear', 
                        'propertycountylandusecode']
    df = prepare.remove_columns(df, cols_to_remove2)
    # fill null values in unitcount with 1 for single unit
    df.unitcnt = df.unitcnt.fillna(value=1)

    # NOTE: if any nulls will be filled with median or mode, or imputed, make sure to split data BEFORE finding median or mode or imputing
        
    # # remove outliers above 50th percentile of upperbound and drop
    # df = prepare.add_upper_outlier_columns(df, k=1.5)
    # zup_drop_index = df[df.taxamount_up_outliers > 5365].index
    # df.drop(zup_drop_index, inplace=True)
    # # remove outliers above 75th percentile of lowerbound and drop
    # df = prepare.add_lower_outlier_columns(df, k=1.5)
    # zlow_drop_index = df[df.taxamount_low_outliers > 9695].index
    # df.drop(zlow_drop_index, inplace=True)

    # drop rows not needed for explore or modeling
    # cols_to_remove3 = [col for col in df if col.endswith('_outliers')]
    # df = prepare.remove_columns(df, cols_to_remove3)
    
    #(new shape = 51827, 26) too many outliers removed with this method!!

    # add features
    df = create_features(df)
    
    # try different method to remove outliers using those with calculated tax rate above 6.6% and below 1%
    df = remove_outliers(df)
    # df shape before drop nulls (70329, 25)

    # drop remaining null vaules for MVP - lot size and land $ are biggest with 7663 each
    df = df.dropna()
    # df shape (62481, 27)

    # split dataset
    target_var = 'logerror'
    X_train, y_train, X_validate, y_validate, X_test, y_test = split(df, target_var)
        
    # remove columns not needed for explore or modeling
    cols_to_remove4 = ['parcelid', 'yearbuilt', 'landtaxvaluedollarcnt', 'regionidzip', 'rawcensustractandblock', 'bathroomcnt', 'regionidcounty']
    X_train = prepare.remove_columns(X_train, cols_to_remove4)
    X_validate = prepare.remove_columns(X_validate, cols_to_remove4)
    X_test = prepare.remove_columns(X_test, cols_to_remove4)
    print(X_train.shape, X_validate.shape, X_test.shape)
    # (34988, 21) (14996, 21) (12497, 21)

    # df is now ready to scale
    X_train_scaled, X_validate_scaled, X_test_scaled = scale_zillow(X_train, X_validate, X_test)
    # drop any columns not scaled from scaled dataframes
    cols_to_remove5 = ['bedroomcnt', 'calculatedfinishedsquarefeet', 'fullbathcnt', 'latitude',
       'longitude', 'lotsizesquarefeet', 'roomcnt', 'unitcnt',
       'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount',
       'propertylandusedesc', 'county', 'age',
       'taxrate', 'structure_dollar_per_sqft', 'land_dollar_per_sqft',
       'bed_bath_ratio', 'fips']
    X_train_scaled = prepare.remove_columns(X_train_scaled, cols_to_remove5)
    X_validate_scaled = prepare.remove_columns(X_validate_scaled, cols_to_remove5)
    X_test_scaled = prepare.remove_columns(X_test_scaled, cols_to_remove5)
    print(X_train_scaled.shape, X_validate_scaled.shape, X_test_scaled.shape)

    # create X_train_explore version with target added back in
    X_train_exp = X_train.copy()
    X_train_exp['logerror'] = y_train.logerror
    return df, X_train, y_train, X_validate, y_validate, X_test, y_test, X_train_scaled, X_validate_scaled, X_test_scaled, X_train_exp


