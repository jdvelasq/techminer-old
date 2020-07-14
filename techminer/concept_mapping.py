def association_analysis(
    X,
    method="MDS",
    n_components=2,
    n_clusters=2,
    linkage="ward",
    x_axis=0,
    y_axis=1,
    figsize=(6, 6),
):

    matplotlib.rc("font", size=11)
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
    clustering.fit(1 - X)
    cluster_dict = {key: value for key, value in zip(X.columns, clustering.labels_)}

    if method == "MDS":
        # Multidimensional scaling
        embedding = MDS(n_components=n_components)
        X_transformed = embedding.fit_transform(X,)

    if method == "CA":
        # Correspondence analysis
        X_transformed = correspondence_matrix(X)

    colors = []
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = pyplot.cm.get_cmap(cmap_name)
        colors += [cmap(0.025 + 0.05 * i) for i in range(20)]

    node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in X.columns]
    max_size = max(node_sizes)
    min_size = min(node_sizes)
    node_sizes = [
        600 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
    ]

    node_colors = [
        cmap(0.2 + 0.80 * cluster_dict[t] / (n_clusters - 1)) for t in X.columns
    ]

    x_axis = X_transformed[:, x_axis]
    y_axis = X_transformed[:, y_axis]

    ax.scatter(
        x_axis, y_axis, s=node_sizes, linewidths=1, edgecolors="k", c=node_colors
    )

    common.ax_expand_limits(ax)

    pos = {term: (x_axis[idx], y_axis[idx]) for idx, term in enumerate(X.columns)}
    common.ax_text_node_labels(
        ax=ax, labels=X.columns, dict_pos=pos, node_sizes=node_sizes
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)

    ax.set_aspect("equal")
    ax.axis("off")
    common.set_ax_splines_invisible(ax)

    fig.set_tight_layout(True)

    return fig


def correspondence_matrix(X):
    """
    """

    matrix = X.values
    grand_total = np.sum(matrix)
    correspondence_matrix = np.divide(matrix, grand_total)
    row_totals = np.sum(correspondence_matrix, axis=1)
    col_totals = np.sum(correspondence_matrix, axis=0)
    independence_model = np.outer(row_totals, col_totals)
    norm_correspondence_matrix = np.divide(correspondence_matrix, row_totals[:, None])
    distances = np.zeros(
        (correspondence_matrix.shape[0], correspondence_matrix.shape[0])
    )
    norm_col_totals = np.sum(norm_correspondence_matrix, axis=0)
    for row in range(correspondence_matrix.shape[0]):
        distances[row] = np.sqrt(
            np.sum(
                np.square(norm_correspondence_matrix - norm_correspondence_matrix[row])
                / col_totals,
                axis=1,
            )
        )
    std_residuals = np.divide(
        (correspondence_matrix - independence_model), np.sqrt(independence_model)
    )
    u, s, vh = np.linalg.svd(std_residuals, full_matrices=False)
    deltaR = np.diag(np.divide(1.0, np.sqrt(row_totals)))
    rowScores = np.dot(np.dot(deltaR, u), np.diag(s))

    return rowScores


#
# Association analysis
#
def __TAB2__(data, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in data.columns if column not in EXCLUDE_COLS])
    #
    left_panel = [
        gui.dropdown(
            desc="Method:",
            options=["Multidimensional scaling", "Correspondence analysis"],
        ),
        gui.dropdown(
            desc="Column:", options=[z for z in COLUMNS if z in data.columns],
        ),
        gui.dropdown(desc="Top by:", options=["Num Documents", "Times Cited",],),
        gui.top_n(),
        gui.normalization(),
        gui.n_components(),
        gui.n_clusters(),
        gui.linkage(),
        gui.x_axis(),
        gui.y_axis(),
        gui.fig_width(),
        gui.fig_height(),
    ]
    # -------------------------------------------------------------------------
    #
    # Logic
    #
    # -------------------------------------------------------------------------
    def server(**kwargs):
        #
        # Logic
        #
        method = {"Multidimensional scaling": "MDS", "Correspondence analysis": "CA"}[
            kwargs["method"]
        ]
        column = kwargs["column"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        normalization = kwargs["normalization"]
        n_components = int(kwargs["n_components"])
        n_clusters = int(kwargs["n_clusters"])
        x_axis = int(kwargs["x_axis"])
        y_axis = int(kwargs["y_axis"])
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        left_panel[8]["widget"].options = list(range(n_components))
        left_panel[9]["widget"].options = list(range(n_components))
        x_axis = left_panel[8]["widget"].value
        y_axis = left_panel[9]["widget"].value

        matrix = co_occurrence_matrix(
            data=data,
            column=column,
            top_by=top_by,
            top_n=top_n,
            normalization=normalization,
            limit_to=limit_to,
            exclude=exclude,
        )

        output.clear_output()
        with output:

            display(
                association_analysis(
                    X=matrix,
                    method=method,
                    n_components=n_components,
                    n_clusters=n_clusters,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    figsize=(width, height),
                )
            )

        return

    ###
    output = widgets.Output()
    return gui.TABapp(left_panel=left_panel, server=server, output=output)

