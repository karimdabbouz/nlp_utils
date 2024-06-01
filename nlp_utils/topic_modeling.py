from nltk.corpus import stopwords
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer



class Bertopic():
    '''
    This class represents a set of text documents and their computed topics.
    It uses BERTopic to compute topics. Embeddings and other options can be passed in at instantiation.
    Topics are computed on instantiation. BERTopic's built-in methods are exposed via the attributes.
    '''

    def __init__(self, embedding_model, documents):
        self.documents = documents
        self.embedding_model = SentenceTransformer(embedding_model)
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        self.hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        self.german_stopwords = stopwords.words('german')
        self.vectorizer_model = CountVectorizer(stop_words=self.german_stopwords)
        self.ctfidf_model = ClassTfidfTransformer()
        self.representation_model =  KeyBERTInspired()
        self.bertopic = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model
        )
        self.topic_model = self.compute_topics()
        self.topic_table = self.bertopic.get_topic_info()
        self.document_table = self.bertopic.get_document_info(self.documents)


    def compute_topics(self):
        '''
        Computes topics using BERTopic.
        Returns the topics and probabilities.
        '''
        topics, probs = self.bertopic.fit_transform(self.documents)
        return topics, probs