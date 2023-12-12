import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    def __init__(self):
        self.embedding = None
        self.finance = None
        self.cluster_labels = None

    def fit(self, embedding, finance, K):
        """
        Fit the model to the embedding data and store the finance data.
        """
        self.embedding = embedding
        self.finance = finance
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=K, random_state=0).fit(embedding)
        self.cluster_labels = kmeans.labels_

    def plot_feature(self, *features):
        """
        Plot the specified features from the finance data.
        If two features are given, plot a scatter plot with cluster color-coding.
        """
        if self.embedding is None or self.finance is None:
            raise ValueError("Data not fitted yet")

        if len(features) == 1:
            feature = features[0]
            if feature not in self.finance.columns:
                raise ValueError(f"Feature {feature} not found in finance data")

            feature_data = self.finance[feature].reindex(self.embedding.index)

            if type(feature_data[0])==type(""):
                
                plt.figure(figsize=(12, 6))
                df = pd.DataFrame({'Cluster': self.cluster_labels, 'Feature': feature_data})
                counts = df.groupby(['Cluster', 'Feature']).size().unstack().fillna(0)
                counts.plot(kind='bar', stacked=True)
                plt.xlabel("Cluster")
                plt.ylabel("Count")
                plt.title(f"Distribution of {feature} by Cluster")
                plt.legend(title=feature, loc='upper left', bbox_to_anchor=(1, 1))
                plt.show()
                
            else:
                plt.figure(figsize=(6, 3))
                sns.boxplot(x=self.cluster_labels, y=feature_data)
                plt.xlabel("Cluster")
                plt.ylabel(feature)
                plt.title(f"Boxplot of {feature} by Cluster")

        elif len(features) == 2:
            feature_x, feature_y = features
            if feature_x not in self.finance.columns or feature_y not in self.finance.columns:
                raise ValueError(f"One of the features {feature_x} or {feature_y} not found in finance data")

            data_x = self.finance[feature_x].reindex(self.embedding.index)
            data_y = self.finance[feature_y].reindex(self.embedding.index)

            plt.figure(figsize=(5, 5))
            sns.scatterplot(x=data_x, y=data_y, hue=self.cluster_labels, palette='rainbow')
            plt.xlabel(feature_x)
            plt.ylabel(feature_y)
            plt.title(f"Scatter Plot of {feature_x} vs {feature_y} by Cluster")
            plt.legend(title='Cluster')

        else:
            raise ValueError("Only 1 or 2 features are allowed")

        plt.show()
        
    def get_ticker_incluster(self,j):
        return [self.embedding.index.values[i] for i in range(len(self.embedding)) if self.cluster_labels[i]==j]
