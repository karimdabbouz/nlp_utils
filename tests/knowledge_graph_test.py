from nlp-utils.knowledge_graph import KnowledgeGraph


def test_knowledge_graph():
    '''
    Tests the KnowledgeGraph class.
    '''
    example_doc = ['Angela Merkel war Bundeskanzlerin von Deutschland. Angela Merkel ist schon lange ein Mitglied der CDU und kommt aus Ostdeutschland. Das Wetter ist schön und als dem Hund eine Katze begegnet, rennt er ihr hinterher. Der Hund isst gerne Lachsfilet.']
    doc_graph = KnowledgeGraph(example_doc, 'DIRECTED')
    print(doc_graph.nodes)
    print(doc_graph.edges)
    doc_graph.visualize_pyvis()


if __name__ == '__main__':
    test_knowledge_graph()
