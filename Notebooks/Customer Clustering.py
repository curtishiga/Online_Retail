import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings

    warnings.filterwarnings(action = 'ignore')
    return np, pd, plt, sns


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    # from sklearn.cluster import AgglomerativeClustering
    # from sklearn.manifold import TSNE
    return KMeans, StandardScaler, silhouette_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Preprocessing
    ## Feature Selection and Justification

    Using the RFM (Recency, Frequency, Monetary) analysis done in another notebook, we perform customer clustering to identify distinct customer segments based on their purchasing behavior. These RFM-derived features are ideal for clustering as they capture the three fundamental dimensions of customer value and behavior.

    Selected Features:
    - **AvgDaysBetweenPurchases**: Frequency-related metric capturing purchase habits. Distinguishes between loyal repeat customers and sporadic buyers.
    - **day_since_last_purchase**: Recency metric measuring how recently a customer made a purchase. Helps identify active vs. churned customers.
    - **TotalSpend**: The total monetary value spent by each customer. This represents the monetary value of a customer and is key to identifying high-value segments.
    """)
    return


@app.cell
def _(pd):
    # Import data
    data = pd.read_excel('../Data/Online Retail_Analysis.xlsx',
                         header = 0,
                         sheet_name = 'RFM')
    return (data,)


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(data):
    # Extract the selected features from the RFM analysis data
    data_features = data[['CustomerID','TotalSpend',
                          'day_since_last_purchase',
                          'AvgDaysBetweenPurchases']]\
                    .dropna()
    return (data_features,)


@app.cell
def _(StandardScaler):
    scaler = StandardScaler()
    return (scaler,)


@app.cell
def _(data_features, pd, scaler):
    data_scaled = pd.DataFrame(scaler.fit_transform(data_features.drop(columns = ['CustomerID'])),
                              columns = data_features.columns)
    return (data_scaled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # K-Means Clustering

    ## Why K-Means Clustering?
    K-Means was selected for several reasons:
    1. **Interpretability**: Creates simple, spherical clusters that are easy to understand and communicate to business stakeholders.4. **Parameter Intuition**: The number of clusters k has clear business meaning (number of customer segments).
    2. **Scalability**: Efficient with our dataset size and number of features.3. **Business Alignment**: Aligns well with RFM segmentation, a proven business methodology.

    **Alternative algorithms considered**:
    - DBSCAN: Better for irregular shapes but less intuitive for business applications
    - Hierarchical Clustering: More computationally expensive, harder to interpret
    """)
    return


@app.cell
def _(np):
    # Set random seed for reproducibility
    # K-Means uses random initialization for centroids, so we set a seed to ensure consistent results across runs
    random_seed_num = 2026
    np.random.seed(random_seed_num)
    return (random_seed_num,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameter Tuning: Elbow Method

    ### What is it?
    The elbow method identifies the optimal number of clusters by analyzing how the within-cluster sum of squares (WCSS) decreases as we increase k.


    ### Why we use it:
    - Helps prevent both underfitting (too few clusters) and overfitting (too many clusters). We look for the "elbow", the point where the WCSS curve changes from steep to flat. Beyond this point, increasing k doesn't significantly improve cluster quality but increases complexity.

    ### How to interpret:
    - Visual method that reveals the "elbow point" where adding more clusters yields diminishing returns

    ### Technical Details:
    **Within-Cluster Sum of Squares (WCSS)**: Measures the compactness of clusters by summing the squared Euclidean distances from each point to its cluster center. Lower WCSS means tighter, more homogeneous clusters.
    """)
    return


@app.cell
def _(KMeans, data_scaled, pd, plt, random_seed_num, sns):
    # Calculate WCSS for cluster range 2-10 to identify the elbow point
    wcss = pd.DataFrame()
    for _i in range(2, 11):
        # Train K-Means model with i clusters
        kmeans_elbow = KMeans(n_clusters=_i, random_state=random_seed_num, n_init=10)
        kmeans_elbow.fit(data_scaled)
        # Store the inertia (WCSS) for each k value
        wcss.loc[_i, 'WCSS'] = kmeans_elbow.inertia_

    # Visualize the elbow curve
    _fig, _ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(wcss, x=wcss.index, y='WCSS', ax=_ax, marker='o')
    plt.title('Elbow Method: WCSS vs Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation:

    The plot shows a sharp decline in WCSS from 2 to 4 clusters, with diminishing returns after 4 clusters. where we capture most of the variance with reasonable model complexity.
    This creates the characteristic "elbow" shape at k=4, indicating that 4 clusters is the optimal point
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameter Tuning: Silhouette Method

    ### What is it?
    The silhouette method evaluates cluster separation and cohesion through a score ranging from -1 to 1.

    ### Why we use it:
    - Directly measures how well-separated clusters are from each other
    - Complements the elbow method with a statistical measure of clustering quality
    - More definitive than the visual elbow method, reducing subjective interpretation

    ### Technical Details:
    **Overall Score**: Average silhouette score across all points indicates overall clustering quality. Higher is better.

    **Silhouette Score**: For each point, measures how similar it is to its own cluster vs. neighboring clusters.
    - Score near 0: Point is on the boundary between clusters
    - Score near +1: Point is well-matched to its cluster
    - Score near -1: Point may be assigned to the wrong cluster
    """)
    return


@app.cell
def _(KMeans, data_scaled, pd, plt, random_seed_num, silhouette_score, sns):
    # Calculate silhouette scores for cluster range 2-10
    silhouette_scores = pd.DataFrame()
    for _i in range(2, 11):
        # Train K-Means model with i clusters
        kmeans_silho = KMeans(n_clusters=_i, random_state=random_seed_num, n_init=10)
        cluster_labels = kmeans_silho.fit_predict(data_scaled)
        # Calculate and store the silhouette score for each k value
        silhouette_scores.loc[_i, 'Silhouette Score'] = silhouette_score(data_scaled, cluster_labels)

    # Visualize the silhouette scores
    _fig, _ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(silhouette_scores, x=silhouette_scores.index, y='Silhouette Score', ax=_ax, marker='o')
    plt.title('Silhouette Analysis: Cluster Separation vs Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation & Decision:

    The silhouette scores peak around 4-5 clusters, with a notable decline after 5. When combined with the elbow analysis, both methods confirm that **k=4** is optimal.

    This decision balances:
    - **Model simplicity**: 4 segments are actionable and memorable for business teams
    - **Overfitting risk**: Avoids unnecessary complexity that 5+ clusters would introduce
    - **Clustering quality**: High silhouette score indicates well-separated clusters

    **Final Decision**:

    Use **k=4 clusters** for the final model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final Modeling

    Based on both the elbow and silhouette analyses, we now train the final K-Means model with **k=4 clusters**
    on the full scaled dataset. This model will segment customers into 4 distinct groups for downstream analysis.
    """)
    return


@app.cell
def _(KMeans, random_seed_num):
    kmean_final = KMeans(n_clusters = 4,
                         random_state = random_seed_num)
    return (kmean_final,)


@app.cell
def _(data_scaled, kmean_final):
    kmean_predict = kmean_final.fit_predict(data_scaled)
    return (kmean_predict,)


@app.cell
def _(data_features, kmean_predict):
    data_features['Cluster Label'] = kmean_predict
    return


@app.cell
def _(data_features):
    data_features.head(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cluster Analysis & Interpretation

    ## Analyzing the Results

    Now that we've identified 4 customer clusters, we examine their characteristics to understand what each segment represents and how to target them effectively.
    """)
    return


@app.cell
def _(data_features):
    # Generate descriptive statistics for each cluster
    data_features.drop(columns = ['CustomerID']).groupby('Cluster Label').describe().transpose()
    return


@app.cell
def _(data_features, mo, plt, sns):
    # Visualize feature distributions by cluster
    mo.md("### Visual Cluster Comparison\nBox plots show the distribution of each feature across the 4 clusters:")
    _fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 8))

    # Total Spend distribution
    sns.boxplot(data_features, x='Cluster Label', y='TotalSpend', ax=ax1, palette='Set2')
    ax1.set_title('Total Spend by Cluster')
    ax1.set_ylabel('Total Spend ($)')

    # Average days between purchases
    sns.boxplot(data_features, x='Cluster Label', y='AvgDaysBetweenPurchases', ax=ax2, palette='Set2')
    ax2.set_title('Average Days Between Purchases by Cluster')
    ax2.set_ylabel('Days')

    # Days since last purchase (Recency)
    sns.boxplot(data_features, x='Cluster Label', y='day_since_last_purchase', ax=ax3, palette='Set2')
    ax3.set_title('Days Since Last Purchase (Recency) by Cluster')
    ax3.set_ylabel('Days')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cluster Segmentation Summary
    The box plots reveal distinct customer segments with clear business implications:

    **Cluster 3 - VIP Customers (High-Value Loyalists)**
    - Characteristics: Highest total spend, shortest time between purchases, most recent activity

    - Business Implication: Core revenue drivers; highest lifetime value; ideal for loyalty programs and premium offerings

    - Action: Priority for retention; exclusive perks and personalized service recommended


    **Cluster 0 - Regular Customers (Loyal But Lower-Spend)**

    - Characteristics: Moderate spend, frequent purchases, reasonably recent activity

    - Business Implication: Stable customer base; opportunity for upselling to higher spend levels

    - Action: Cross-sell strategies; bundle offers to increase average order value

    **Cluster 2 - Win-Back Customers (Returned Churned Customers)**

    - Characteristics: Low spend, long time between purchases but have recently returned

    - Business Implication: Customers who left but came back; fragile re-engagement window

    - Action: Special re-engagement offers; customer satisfaction surveys to understand previous churn


    **Cluster 1 - At-Risk Customers (One-Time/Churned)**
    - Characteristics: Low spend, very long time since last purchase (high recency), low purchase frequency

    - Action: Win-back campaigns; understand reasons for churn; consider removing from active marketing if non-responsive

    - Business Implication: Lost customers; minimal current revenue contribution

    ### Summary Table:
    | Cluster | Label | Value | Loyalty | Status | Strategy |
    |---------|-------|-------|---------|--------|----------|
    | 3 | VIP | High | Very High | Active | Retention, Premium |
    | 0 | Regular | Moderate | High | Active | Upsell, Cross-sell |
    | 2 | Win-Back | Low | Returning | At-Risk | Re-engage, Satisfy |
    | 1 | At-Risk | Low | Very Low | Churned | Win-back, Analyze |

    This segmentation provides a data-driven foundation for targeted marketing campaigns, personalized customer experiences, and strategic resource allocation.
    """)
    return


if __name__ == "__main__":
    app.run()
