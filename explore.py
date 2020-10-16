import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import os


def plot_variable_pairs(df):
    g = sns.PairGrid(df) 
    g.map_diag(sns.distplot)
    g.map_offdiag(sns.regplot)

def months_to_years(tenure_months, df):
    df['tenure_years'] = round(tenure_months/12, 0)
    return df

# def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):
#     plt.rc('font', size=13)
#     plt.rc('figure', figsize=(13, 7))
#     plt.subplot(311)
#     sns.boxplot(data=df, y=continuous_var, x=categorical_var)
#     plt.subplot(312)
#     sns.violinplot(data=df, y=continuous_var, x=categorical_var)
#     plt.subplot(313)
#     sns.swarmplot(data=df, y=continuous_var, x=categorical_var)
#     plt.tight_layout()
#     plt.show()

def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):
    plt.rc('font', size=13)
    plt.rc('figure', figsize=(13, 7))
    sns.boxplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()   
    sns.violinplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()
    sns.swarmplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()



def elbow_plot(cluster_vars):
    # elbow method to identify good k for us
    ks = range(2,20)
    
    # empty list to hold inertia (sum of squares)
    sse = []

    # loop through each k, fit kmeans, get inertia
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_train_scaled[cluster_vars])
        # inertia
        sse.append(kmeans.inertia_)

    print(pd.DataFrame(dict(k=ks, sse=sse)))

    # plot k with inertia
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('Elbow method to find optimal k')
    plt.show()

####### elbow_plot(cluster_vars = area_vars)


def run_kmeans(k, cluster_vars, cluster_col_name):
    # create kmeans object
    kmeans = KMeans(n_clusters = k, random_state = 13)
    kmeans.fit(X_train_scaled[cluster_vars])
    # predict and create a dataframe with cluster per observation
    train_clusters = \
        pd.DataFrame(kmeans.predict(X_train_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_train.index)
    
    return train_clusters, kmeans

####### train_clusters, kmeans = run_kmeans(k=6, 
                                    # cluster_vars = ['latitude', 
                                    #                 'longitude', 
                                    #                 'age'], 
                                    # cluster_col_name = 'area_cluster')


def get_centroids(cluster_vars, cluster_col_name):
    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroids = pd.DataFrame(kmeans.cluster_centers_, 
             columns=centroid_col_names).reset_index().rename(columns={'index': cluster_col_name})
    
    return centroids

######### centroids = get_centroids(cluster_vars, cluster_col_name='size_cluster')


def add_to_train(cluster_col_name):
    # concatenate cluster id
    X_train2 = pd.concat([X_train, train_clusters], axis=1)

    # join on clusterid to get centroids
    X_train2 = X_train2.merge(centroids, how='left', 
                            on=cluster_col_name).\
                        set_index(X_train.index)
    
    # concatenate cluster id
    X_train_scaled2 = pd.concat([X_train_scaled, train_clusters], 
                               axis=1)

    # join on clusterid to get centroids
    X_train_scaled2 = X_train_scaled2.merge(centroids, how='left', 
                                          on=cluster_col_name).\
                            set_index(X_train.index)
    
    return X_train2, X_train_scaled2

####### X_train, X_train_scaled = add_to_train(cluster_col_name = 'size_cluster')

