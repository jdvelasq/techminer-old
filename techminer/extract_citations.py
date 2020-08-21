import pandas as pd


def extract_citations(input_file="techminer.csv", output_file="techminer.csv"):

    data = pd.read_csv(input_file)
    data["Cited_References"] = [[] for _ in range(len(data))]
    for i_index, _ in enumerate(data.Title):

        title = data.Title[i_index].lower()

        for j_index, references in enumerate(data.References.tolist()):

            if pd.isna(references) is False:
                if title in references.lower():
                    data.at[i_index, "Cited_References"] += [data.Document[j_index]]

    data["Cited_References"] = data.Cited_References.map(
        lambda w: pd.NA if len(w) == 0 else w
    )
    data["Cited_References"] = data.Cited_References.map(
        lambda w: ";".join(w), na_action="ignore"
    )
    data.to_csv(output_file, index=False)
