import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
from sklearn.cluster import KMeans
import summarize
import prepare

def summary(df):
    # print summary info then remove generated columns
    df = summarize.df_summary(df)
    cols_to_remove3 = ['null_count', 'pct_null', ]
    df = prepare.remove_columns(df, cols_to_remove3)
    return df

def plot_variable_pairs(df):
    g = sns.PairGrid(df) 
    g.map_diag(sns.distplot)
    g.map_offdiag(sns.regplot)


def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):
    plt.rc('font', size=13)
    plt.rc('figure', figsize=(13, 7))
    sns.boxplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()   
    sns.violinplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()
    sns.swarmplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()

def pearson(continuous_var1, continuous_var2):
    alpha = .05
    r, p = stats.pearsonr(continuous_var1, continuous_var2)
    print('r=', r)
    print('p=', p)
    if p < alpha:
        print("We reject the null hypothesis")
        print(f'p     = {p:.4f}')
    else:
        print("We fail to reject the null hypothesis")
    return r, p

def elbow_plot(X_train_scaled, cluster_vars):
    # elbow method to identify good k for us, originally used range (2,20), changed for presentation
    ks = range(2,16)
    
    # empty list to hold inertia (sum of squares)
    sse = []

    # loop through each k, fit kmeans, get inertia
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_train_scaled[cluster_vars])
        # inertia
        sse.append(kmeans.inertia_)
    # print out was used for determining cutoff, commented out for presentation
    # print(pd.DataFrame(dict(k=ks, sse=sse)))

    # plot k with inertia
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('Elbow method to find optimal k')
    plt.show()

####### elbow_plot(X_train_scaled, cluster_vars = area_vars)


def run_kmeans(X_train_scaled, X_train, cluster_vars, k, cluster_col_name):
    # create kmeans object
    kmeans = KMeans(n_clusters = k, random_state = 13)
    kmeans.fit(X_train_scaled[cluster_vars])
    # predict and create a dataframe with cluster per observation
    train_clusters = \
        pd.DataFrame(kmeans.predict(X_train_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_train.index)
    
    return train_clusters, kmeans

####### train_clusters, kmeans = run_kmeans(X_train_scaled, X_train, k, cluster_vars, cluster_col_name)

def kmeans_transform(X_scaled, kmeans, cluster_vars, cluster_col_name):
    kmeans.transform(X_scaled[cluster_vars])
    trans_clusters = \
        pd.DataFrame(kmeans.predict(X_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_scaled.index)
    
    return trans_clusters

####### trans_clusters = kmeans_transform(X_scaled, kmeans, cluster_vars, cluster_col_name)


def get_centroids(kmeans, cluster_vars, cluster_col_name):
    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroids = pd.DataFrame(kmeans.cluster_centers_, 
             columns=centroid_col_names).reset_index().rename(columns={'index': cluster_col_name})
    
    return centroids

######### centroids = get_centroids(kmeans, cluster_vars, cluster_col_name)


def add_to_train(X_train, train_clusters, X_train_scaled, centroids, cluster_col_name):
    # concatenate cluster id
    X_train = pd.concat([X_train, train_clusters], axis=1)

    # join on clusterid to get centroids
    X_train = X_train.merge(centroids, how='left', 
                            on=cluster_col_name).\
                        set_index(X_train.index)
    
    # concatenate cluster id
    X_train_scaled = pd.concat([X_train_scaled, train_clusters], 
                               axis=1)

    # join on clusterid to get centroids
    X_train_scaled = X_train_scaled.merge(centroids, how='left', 
                                          on=cluster_col_name).\
                            set_index(X_train.index)
    
    return X_train, X_train_scaled

####### X_train, X_train_scaled = add_to_train(X_train, train_clusters, X_train_scaled, centroids, cluster_col_name)

