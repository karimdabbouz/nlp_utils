import stanza, logging
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from enum import Enum, auto


class GraphType(Enum):
    DIRECTED = auto()
    UNDIRECTED = auto()


class KnowledgeGraph():
    '''
    This class represents a text document for information retrieval using graph objects.
    It can create different kinds of knowledge graphs and visualize them in multiple ways.
    '''
    def __init__(self, document, graph_type: GraphType):
        self.document = document
        self.stanza_sentences = self.create_stanza_sentences()
        self.graph_type = graph_type
        self.edges, self.nodes, self.labels, self.edge_labels = self.initialize_graph()
        
    
    def create_stanza_sentences(self):
        '''
        Instantiates this object with the document being passed to the class.
        '''
        logging.getLogger('stanza').setLevel(logging.WARNING)
        stanza_pipeline = stanza.Pipeline('de')
        stanza_doc = stanza_pipeline(self.document)
        return stanza_doc.sentences


    def initialize_graph(self):
        '''
        Creates a graph depending on the graph_type and initilializes nodes, edges and labels accordingly.
        '''
        if self.graph_type == 'DIRECTED':
            return self.create_graph_nsubj_obj_directed()
        elif self.graph_type == 'UNDIRECTED':
            return self.create_graph_nsubj_obj()

    
    def create_graph_nsubj_obj_directed(self):
        '''
        Creates a directed graph and graph labels for all subject-object relationships connected with a verb.
        '''
        edges = []
        nodes = []
        labels = []
        for sentence in self.stanza_sentences:
                for word in sentence.words:
                    if (word.pos == 'NOUN' or word.pos == 'PROPN') and word.deprel == 'nsubj' and sentence.words[word.head - 1].pos == 'VERB':
                        for x in sentence.words:
                            if (x.pos == 'NOUN' or x.pos == 'PROPN') and x.id != word.id and sentence.words[x.head - 1].id == sentence.words[word.head - 1].id and x.deprel == 'obj':
                                if (word.lemma, x.lemma) not in edges or (x.lemma, word.lemma) not in edges:
                                    nodes.append(word.lemma)
                                    nodes.append(x.lemma)
                                    edges.append((word.lemma, x.lemma))
                                    labels.append(sentence.words[x.head - 1].lemma)
        edge_labels = {v: labels[i] for i, v in enumerate(edges)}
        return edges, nodes, labels, edge_labels


    def create_graph_nsubj_obj(self):
        '''
        Creates a non-directed graph and graph labels for all subject-object relationsships connected with a verb.
        '''
        edges = []
        nodes = []
        labels = []
        for sentence in self.stanza_sentences:
            for word in sentence.words:
                if (word.pos == 'NOUN' or word.pos == 'PROPN') and word.deprel == 'nsubj' and sentence.words[word.head - 1].pos == 'VERB':
                    for x in sentence.words:
                        if (x.pos == 'NOUN' or x.pos == 'PROPN') and x.id != word.id and sentence.words[x.head - 1].id == sentence.words[word.head - 1].id and x.deprel == 'obj':
                            nodes.append(word.lemma)
                            nodes.append(x.lemma)
                            edges.append((word.lemma, x.lemma))
                            labels.append(sentence.words[x.head - 1].lemma)
        edge_labels = {v: labels[i] for i, v in enumerate(edges)}
        return edges, nodes, labels, edge_labels


    def visualize_pyvis(self):
        '''
        Visualizes the graph. Directed or undirected depending on graph_type.
        Visualization opens in a new browser tab.
        '''
        if self.graph_type == 'DIRECTED':
            net = nx.DiGraph()
            directed = True
        elif self.graph_type == 'UNDIRECTED':
            net = nx.Graph()
            directed = False
        min_size = 10
        max_size = 40
        max_occ = max(Counter(self.nodes).values())
        min_occ = min(Counter(self.nodes).values())
        net.add_nodes_from(self.nodes)
        net.add_edges_from(self.edges)
        for edge, label in self.edge_labels.items():
            net.edges[edge]['label'] = label
        node_sizes = {key: min_size + (max_size - min_size) * (value - min_occ) / (max_occ - min_occ) for key, value in Counter(self.nodes).items()}
        nx.set_node_attributes(net, node_sizes, 'size')
        nt = Network('1000px', '100%', directed=directed)
        nt.from_nx(net)
        nt.show('nx.html', notebook=False)