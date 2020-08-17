import pandas as pd
import techminer.common as cmn

from sklearn.cluster import (
    AgglomerativeClustering,
    AffinityPropagation,
    Birch,
    DBSCAN,
    FeatureAgglomeration,
    KMeans,
    MeanShift,
)


def clustering(
    X,
    method,
    n_clusters,
    affinity,
    linkage,
    random_state,
    top_n,
    name_prefix="Cluster {}",
):

    X = X.copy()

    #
    # 1.-- Compute cluster labels
    #
    labels = None

    if method == "Affinity Propagation":
        labels = AffinityPropagation(random_state=int(random_state)).fit_predict(1 - X)
        n_clusters = len(set(labels))

    if method == "Agglomerative Clustering":
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, affinity=affinity, linkage=linkage
        ).fit_predict(1 - X)

    if method == "Birch":
        labels = Birch(n_clusters=n_clusters).fit_predict(1 - X)

    if method == "DBSCAN":
        labels = DBSCAN().fit_predict(1 - X)
        n_clusters = len(set(labels))

    #  if self.clustering_method == "Feature Agglomeration":
    #      m = FeatureAgglomeration(
    #          n_clusters=self.n_clusters, affinity=self.affinity, linkage=self.linkage
    #      ).fit(1 - X)
    #      labels =

    if method == "KMeans":
        labels = KMeans(
            n_clusters=n_clusters, random_state=int(random_state)
        ).fit_predict(1 - X)

    if method == "Mean Shift":
        labels = MeanShift().fit_predict(1 - X)
        n_clusters = len(set(labels))

    #
    # 2.-- Cluster memberships
    #
    M = pd.DataFrame({"Cluster": labels}, index=X.index)
    num_rows = top_n if top_n is not None else len(X.index)
    cluster_members = pd.DataFrame(
        pd.NA, columns=range(n_clusters), index=range(num_rows)
    )
    for i_cluster in range(n_clusters):
        members = M[M.Cluster == i_cluster]
        if top_n is not None:
            members = cmn.sort_axis(
                data=members, num_documents=True, axis=0, ascending=False
            )
            members = members.head(top_n)
        cluster_members.at[0 : len(members) - 1, i_cluster] = members.index
    cluster_members = cluster_members

    cluster_members.columns = [
        name_prefix.format(i_cluster) for i_cluster in range(n_clusters)
    ]

    row_ids = []
    for row in cluster_members.iterrows():
        if any([not pd.isna(a) for a in row[1]]):
            row_ids.append(row[0])

    cluster_members = cluster_members.loc[row_ids, :]
    cluster_members = cluster_members.applymap(lambda w: "" if pd.isna(w) else w)

    #
    # 3.-- Cluster centres
    #
    X["CLUSTER"] = labels
    cluster_centers = X.groupby("CLUSTER").mean()
    X.pop("CLUSTER")

    #
    # 4.-- Cluster names
    #
    cluster_names = cluster_members.loc[cluster_members.index[0], :].tolist()

    return n_clusters, labels, cluster_members, cluster_centers, cluster_names
