def counters_to_node_sizes(x):
    node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in x]
    max_size = max(node_sizes)
    min_size = min(node_sizes)
    node_sizes = [
        500 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
    ]
    return node_sizes
