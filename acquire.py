import pandas as pd
import numpy as np
import os
from env import host, user, password

#################### Acquire ##################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


############# Zillow Clustering ########  

def cluster_zillow_data():
    '''
    This function reads the  zillow data from the Codeup db into a df,
    write it to a csv file, and returns the df. 
    '''
    sql_query = '''
                select * from properties_2017
                join (select id, logerror, pid, tdate from predictions_2017 pred_2017
                join (SELECT parcelid as pid, Max(transactiondate) as tdate FROM predictions_2017 GROUP BY parcelid) as sq1
                on (pred_2017.parcelid = sq1.pid and pred_2017.transactiondate = sq1.tdate)) as sq2
                on (properties_2017.parcelid = sq2.pid)
                left join airconditioningtype using (airconditioningtypeid)
                left join architecturalstyletype using (architecturalstyletypeid)
                left join buildingclasstype using (buildingclasstypeid)
                left join heatingorsystemtype using (heatingorsystemtypeid)
                left join propertylandusetype using (propertylandusetypeid)
                left join storytype using (storytypeid)
                left join typeconstructiontype using (typeconstructiontypeid)
                left join unique_properties using (parcelid)
                where latitude is not null and longitude is not null
                AND propertylandusetypeid IN ('260', '261', '262', '263', '264', '266', '268', '273', '274', '275', '276', '279');
                '''
    df = pd.read_sql(sql_query, get_connection('zillow'))
    df.to_csv('zillow_cluster_df.csv')
    return df


def get_zillow_cluster_data(cached=False):
    '''
    This function reads in zillow customer data from Codeup database if cached == False 
    or if cached == True reads in mall customer df from a csv file, returns df
    '''
    if cached or os.path.isfile('zillow_cluster_df.csv') == False:
        df = cluster_zillow_data()
    else:
        df = pd.read_csv('zillow_cluster_df.csv', index_col=0)
    return df

# full shape of dataframe is (71,789, 70)