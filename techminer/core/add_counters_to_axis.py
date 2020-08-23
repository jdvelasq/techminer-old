import numpy as np


from techminer.core.explode import explode


def add_counters_to_axis(X, axis, data, column):

    X = X.copy()
    data = data.copy()
    data["Num_Documents"] = 1
    m = (
        explode(data[[column, "Num_Documents", "Times_Cited", "ID"]], column)
        .groupby(column, as_index=True)
        .agg({"Num_Documents": np.sum, "Times_Cited": np.sum,})
    )
    n_Num_Documents = int(np.log10(m["Num_Documents"].max())) + 1
    n_Times_Cited = int(np.log10(m["Times_Cited"].max())) + 1
    fmt = "{} {:0" + str(n_Num_Documents) + "d}:{:0" + str(n_Times_Cited) + "d}"
    new_names = {
        key: fmt.format(key, int(nd), int(tc))
        for key, nd, tc in zip(m.index, m.Num_Documents, m.Times_Cited)
    }
    if axis == 0:
        X.index = [new_names[t] for t in X.index]
    elif axis == 1:
        X.columns = [new_names[t] for t in X.columns]
    else:
        raise NameError("Invalid axis value:" + str(axis))

    return X
