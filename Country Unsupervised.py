import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("29-country_data.csv")

print(df.info())
print(df.head(25))
print(df.describe())

print(df.sort_values(by = "gdpp", ascending = False))

# all histograms

import math

def plot_all_histograms(df, title_prefix=""):
    num_cols = df.select_dtypes(include = [np.number]).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    plt.figure(figsize = (5 * n_cols, 4 * n_rows ))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"{title_prefix}{col}")
        plt.xlabel("")
        plt.ylabel("")

    plt.tight_layout()
    plt.show()

plot_all_histograms(df)

sns.heatmap(df.corr(numeric_only = True), annot = True)
plt.show()

df2 = df.drop("country", axis = 1)
print(df2)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df2 = scaler.fit_transform(df2)

df2_cols = scaler.get_feature_names_out()

df2 = pd.DataFrame(df2, columns = df2_cols)

print(df2)

plot_all_histograms(df2)

from sklearn.decomposition import PCA
pca = PCA()

pca_df2 = pd.DataFrame(pca.fit_transform(df2))

print(pca_df2)

print(pca.explained_variance_)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.ylabel("Variance Covered")
plt.title("Variance Covered")
plt.show()


# aslında kümülatif gitmesi yani toplayarak girmesi gerekiyor bunun için plt.step(adım, işlem) np.cumsum ekleriz

plt.step(list(range(1,10)), np.cumsum(pca.explained_variance_ratio_))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.ylabel("Variance Covared")
plt.title("Variance Covered")
plt.show()


# 3 de neredeyse verinin yüzde 90'ını alıyoruz yani pca yi 3 kolon olarak alabiliriz 

pca_df2 = pca_df2.drop(columns = [3,4,5,6,7,8])

print(pca_df2)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

wcss = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=23)
    kmeans.fit(pca_df2)
    wcss.append(kmeans.inertia_)

print(wcss)
   
plt.plot(range(1,11),wcss)
plt.xticks(range(1,11))
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
   
# bu grafik ile çokta anlamlı bir çıkarım elde edemedik 

model = KMeans(n_clusters=3, random_state=23)
model.fit(pca_df2)

labels = model.labels_

print(silhouette_score(pca_df2, labels))

df["Class"] = labels

print(df[df["country"] == "Singapore"]) # düzeyi kontrol etmek için

fig, ax = plt.subplots(nrows = 1, ncols= 2 , figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data=df, x="Class", y="child_mort")
plt.title("child_mort vs class")

plt.subplot(1,2,2)
sns.boxplot(data=df, x="Class", y="income")
plt.title("income vs class")

plt.show()

import plotly.express as px

pca_df2.insert(0, column= "Country", value = df['country'])

pca_df2['Class'] = labels

# Önce sütunun veri tipini değiştiriyoruz

pca_df2["Class"] = pca_df2["Class"].astype("object")

# Sonra atama işlemlerini yapıyoruz

pca_df2.loc[pca_df2["Class"] == 0, "Class"] = "No Budget Needed"
pca_df2.loc[pca_df2["Class"] == 1, "Class"] = "Budget Needed"
pca_df2.loc[pca_df2["Class"] == 2, "Class"] = "In Between"


fig = px.choropleth(
    pca_df2[['Country', 'Class']],
    locationmode = "country names",
    locations = "Country",
    title = "Needed Budget by Country",
    color = pca_df2['Class'],
    color_discrete_map= {
                        "Budget Needed" : "Red",
                        "In Between" : "Yellow",
                        "No Budget Needed": "Green"
    })
fig.update_geos(fitbounds = "locations", visible = True)
fig.show()

# Kümeleme algoritmaları metin verisi kabul etmez, bu yüzden string sütunları atıyoruz
pca_df2 = pca_df2.drop(columns=["Country", "Class"])

model = KMeans(n_clusters=3, random_state=23)
model.fit(pca_df2)

labels = model.labels_
print("Kmeans SS with PCA",silhouette_score(pca_df2, labels))

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3)

y_hc = hc.fit_predict(pca_df2)

print(f"HC SS with PCA: {silhouette_score(pca_df2, y_hc)}")

from sklearn.cluster import DBSCAN, HDBSCAN

db = DBSCAN()

db.fit(pca_df2)

dblabel = db.labels_

print("DB SS with PCA", silhouette_score(pca_df2, dblabel))

hdb = HDBSCAN()

hdb.fit(pca_df2)

hdblabel = hdb.labels_

print("HDB SS PCA",silhouette_score(pca_df2, hdblabel))

eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
min_samples_values = [4, 5, 6]

results = []

for eps in eps_values:
    for min_samples_val in min_samples_values:

        db = DBSCAN(eps=eps, min_samples=min_samples_val).fit(pca_df2)

        labels = db.labels_

        if len(set(labels)) <= 1:
            continue

        silhouette = silhouette_score(pca_df2, labels)
        results.append(
            {
                "eps" : eps,
                "min_samples" : min_samples_val,
                "Silhouette" : silhouette,
                "n_clusters" : len(set(labels)) - (1 if -1 in labels else 0)
            }
        )

results_df = pd.DataFrame(results).sort_values(by="Silhouette", ascending=False)
print(results_df)

# ... HDBSCAN Kodu başlangıcı ...
from sklearn.cluster import HDBSCAN

min_cluster_sizes = [3, 5, 7, 10]
min_samples_list = [None, 3, 5, 7] 

results_hdbscan = []

for min_cluster in min_cluster_sizes:
    for min_sample in min_samples_list:
       
        hdb = HDBSCAN(min_cluster_size=min_cluster, min_samples=min_sample).fit(pca_df2)

        labels = hdb.labels_

        if len(set(labels)) <= 1:
            continue

        silhouette = silhouette_score(pca_df2, labels)
        results_hdbscan.append(
            {
                "min_cluster" : min_cluster, 
                "min_sample" : min_sample,
                "Silhouette" : silhouette,
                "n_clusters" : len(set(labels)) - (1 if -1 in labels else 0)
            }
        )

results_hdbscan_df = pd.DataFrame(results_hdbscan).sort_values(by="Silhouette", ascending=False)
print(results_hdbscan_df)

# CLUSTERING WITHOUT PCA

df = df.drop(["country", "Class"], axis = 1)

print(df)
scaler = MinMaxScaler()

df = scaler.fit_transform(df)

df_cols = scaler.get_feature_names_out()

df = pd.DataFrame(df, columns = df_cols)

df_wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=23)
    kmeans.fit(df)
    df_wcss.append(kmeans.inertia_)

print(df_wcss)

plt.plot(range(1,11), df_wcss)
plt.xticks(range(1,11))
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

from kneed import KneeLocator

kl = KneeLocator(range(1,11), df_wcss, curve="convex", direction="decreasing")

print(kl.elbow)

model = KMeans(n_clusters=3, random_state=23)
model.fit(df)

labels = model.labels_

print("Kmeans Silhouette Score: ",silhouette_score(df, labels))

hc = AgglomerativeClustering(n_clusters=3)

y_hc = hc.fit_predict(df)

print(f"HC Silhouette Score: {silhouette_score(df, y_hc)}")

from sklearn.cluster import DBSCAN, HDBSCAN

db = DBSCAN()

db.fit(df)

dblabel = db.labels_

print("DB Silhouette Score: ", silhouette_score(df, dblabel))

hdb = HDBSCAN()

hdb.fit(df)

hdblabel = hdb.labels_

print("HDB Silhouette Score: ",silhouette_score(df, hdblabel))

# HYPERPARAMETER TUNING

eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
min_samples_values = [4, 5, 6]

results = []

for eps in eps_values:
    for min_samples_val in min_samples_values:

        db = DBSCAN(eps=eps, min_samples=min_samples_val).fit(df)

        labels = db.labels_

        if len(set(labels)) <= 1:
            continue

        silhouette = silhouette_score(df, labels)
        results.append(
            {
                "eps" : eps,
                "min_samples" : min_samples_val,
                "Silhouette" : silhouette,
                "n_clusters" : len(set(labels)) - (1 if -1 in labels else 0)
            }
        )

results_df = pd.DataFrame(results).sort_values(by="Silhouette", ascending=False)
print(results_df)

# ... HDBSCAN Kodu başlangıcı ...
from sklearn.cluster import HDBSCAN

min_cluster_sizes = [3, 5, 7, 10]
min_samples_list = [None, 3, 5, 7] 

results_hdbscan = []

for min_cluster in min_cluster_sizes:
    for min_sample in min_samples_list:
    
        hdb = HDBSCAN(min_cluster_size=min_cluster, min_samples=min_sample).fit(df)

        labels = hdb.labels_

        if len(set(labels)) <= 1:
            continue

        silhouette = silhouette_score(df, labels)
        results_hdbscan.append(
            {
                "min_cluster" : min_cluster, 
                "min_sample" : min_sample,
                "Silhouette" : silhouette,
                "n_clusters" : len(set(labels)) - (1 if -1 in labels else 0)
            }
        )

results_hdbscan_df = pd.DataFrame(results_hdbscan).sort_values(by="Silhouette", ascending=False)
print(results_hdbscan_df)
