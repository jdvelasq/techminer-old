"""
TechMiner.Maps
==================================================================================================




"""
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from . import DataFrame


class Map:
    def __init__(self):
        self._graph = nx.Graph()

    def _compute_graph_layout(self):
        path_length = nx.shortest_path_length(self._graph)
        distances = pd.DataFrame(index=self._graph.nodes(), columns=self._graph.nodes())
        for row, data in path_length:
            for col, dist in data.items():
                distances.loc[row, col] = dist
        distances = distances.fillna(distances.max().max())
        return nx.kamada_kawai_layout(self._graph, dist=distances.to_dict())

    def add_node(self, node_for_adding, **attr):
        """Adds a node to a current map.
        """
        self._graph.add_node(node_for_adding, **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        """Adds a bunch of nodes to a current map.
        """
        self._graph.add_nodes_from(self, nodes_for_adding, **attr)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Add an edge to a current map.
        """
        self._graph.add_edge(self, u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        """Adds a bunch of edges to a current map.
        """
        self._graph().add_edges_from(self, ebunch_to_add, **attr)

    def ocurrence_map(
        self,
        terms,
        docs,
        edges,
        label_terms,
        label_docs,
        term_props={},
        doc_props={},
        edge_props={},
        label_term_props={},
        label_docs_props={},
    ):
        """Cluster map for ocurrence and co-ocurrence matrices.
    
        >>> terms = ["A", "B", "C", "D"]
        >>> docs = ["doc#0", "doc#1", "doc#2", "doc#3", "doc#4", "doc#5"]
        >>> edges = [
        ...     ("A", "doc#0"),
        ...     ("A", "doc#1"),
        ...     ("B", "doc#1"),
        ...     ("A", "doc#2"),
        ...     ("B", "doc#2"),
        ...     ("C", "doc#2"),
        ...     ("B", "doc#3"),
        ...     ("B", "doc#4"),
        ...     ("D", "doc#4"),
        ...     ("D", "doc#5"),
        ... ]
        >>> label_docs = {
        ...     "doc#0": 2,
        ...     "doc#1": 1,
        ...     "doc#2": 1,
        ...     "doc#3": 1,
        ...     "doc#4": 1,
        ...     "doc#5": 1,
        ... }
        >>> label_terms = {
        ...     "A": "Author A",
        ...     "B": "Author B",
        ...     "C": "Author C",
        ...     "D": "Author D",
        ... }
        >>> nxmap = Map()
        >>> nxmap.ocurrence_map(
        ...     terms,
        ...     docs,
        ...     edges,
        ...     label_terms,
        ...     label_docs,
        ...     term_props={"node_color": "red"},
        ...     label_docs_props={"font_color": "lightblue"},
        ...     label_term_props=dict(ma="left", rotation=0, fontsize=10, disp=3, bbox=None),
        ... )        
        >>> plt.savefig('guide/images/network_occurrence_map.png')
        
        .. image:: images/network_occurrence_map.png
            :width: 600px
            :align: center

        >>> import pandas as pd
        >>> x = [ 'A', 'A', 'A,B', 'B', 'A,B,C', 'D', 'B,D']
        >>> df = pd.DataFrame(
        ...    {
        ...       'Authors': x,
        ...       'ID': list(range(len(x))),
        ...    }
        ... )
        >>> df
          Authors  ID
        0       A   0
        1       A   1
        2     A,B   2
        3       B   3
        4   A,B,C   4
        5       D   5
        6     B,D   6
        >>> nxmap = Map()
        >>> dic1 = DataFrame(df).occurrence_map(column='Authors')
        >>> dic2 = dict(
        ...     term_props={"node_color": "red"}, 
        ...     label_docs_props={"font_color": "lightblue"}, 
        ...     label_term_props=dict(ma="left", rotation=0, fontsize=10, disp=3, bbox=None)
        ... )
        >>> kwargs = {**dic1, **dic2}
        >>> kwargs
        {'terms': ['A', 'B', 'C', 'D'], 'docs': ['doc#0', 'doc#1', 'doc#2', 'doc#3', 'doc#4', 'doc#5'], 'edges': [('A', 'doc#0'), ('A', 'doc#1'), ('B', 'doc#1'), ('A', 'doc#2'), ('B', 'doc#2'), ('C', 'doc#2'), ('B', 'doc#3'), ('B', 'doc#4'), ('D', 'doc#4'), ('D', 'doc#5')], 'label_terms': {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}, 'label_docs': {'doc#0': 2, 'doc#1': 1, 'doc#2': 1, 'doc#3': 1, 'doc#4': 1, 'doc#5': 1}, 'term_props': {'node_color': 'red'}, 'label_docs_props': {'font_color': 'lightblue'}, 'label_term_props': {'ma': 'left', 'rotation': 0, 'fontsize': 10, 'disp': 3, 'bbox': None}}
        >>> nxmap.ocurrence_map(**kwargs)
        >>> plt.savefig('guide/images/network_occurrence_map_1.png')

        .. image:: images/network_occurrence_map_1.png
            :width: 600px
            :align: center
        """
        plt.clf()

        self._graph.clear()
        self._graph.add_nodes_from(terms)
        self._graph.add_nodes_from(docs)
        self._graph.add_edges_from(edges)

        layout = self._compute_graph_layout()

        nx.draw_networkx_nodes(self._graph, pos=layout, nodelist=terms, **term_props)
        nx.draw_networkx_nodes(self._graph, pos=layout, nodelist=docs, **doc_props)
        nx.draw_networkx_edges(self._graph, pos=layout, **edge_props)
        nx.draw_networkx_labels(
            self._graph, pos=layout, labels=label_docs, **label_docs_props
        )
        self.draw_network_labels(pos=layout, labels=label_terms, **label_term_props)

        plt.axis("off")

    def draw_network_labels(self, pos, labels, disp=1, fontdict=None, **kwargs):

        x_left, x_right = plt.xlim()
        y_left, y_right = plt.ylim()
        delta_x = (x_right - x_left) * disp / 100
        delta_y = (y_right - y_left) * disp / 100

        default = dict(
            fontsize=12,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="square", ec="lightgray", fc="white",),
        )
        props = {**default, **kwargs}

        for node in self._graph.nodes:
            if node in labels:
                x_pos, y_pos = pos[node]
                plt.text(
                    x_pos + delta_x, y_pos + delta_y, labels[node], **props,
                )
