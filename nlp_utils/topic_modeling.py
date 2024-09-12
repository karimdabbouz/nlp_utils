from nltk.corpus import stopwords
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import make_scorer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
import numpy as np
import collections



class Bertopic():
    '''
    This class represents a set of text documents and their computed topics.
    It uses BERTopic to compute topics. Embeddings and other options can be passed in at instantiation.
    Topics are computed on instantiation. BERTopic's built-in methods are exposed via the attributes.
    '''

    def __init__(self, embedding_model, min_topic_size, documents):
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
            representation_model=self.representation_model,
            min_topic_size=min_topic_size
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



class SklearnClustering():
    '''
    This class creates clusters from transformer embeddings using either K-means or HDBSCAN.
    You can either define the num of clusters yourself or let it choose the best-scoring params.
    :param str algorithm: The clustering algorithm - "kmeans" (will add more later)
    :param np.array embeddings: The embeddings in shape (num_docs, num_dimensions)
    :param list documents: The list of documents used to create the embeddings
    :param boolean auto_score: Set to True to auto-choose the best scoring params using GridSearchCv
    :param scrorer: The scoring method to use in GridSearchCV
    :param int n_clusters: The number of clusters to use in Kmeans or the min number of clusters in HDBSCAN (optional in mode auto_score)
    :param int n_samples: The min number of samples to use in HDBSCAN (optional in mode auto_score)
    '''
    def __init__(self, algorithm, embeddings, documents, auto_score=False, scorer=None, n_clusters=None, n_samples=None):
        self.auto_score = auto_score
        self.algorithm = algorithm
        self.scorer = scorer
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.embeddings = embeddings
        self.documents = documents
        self.result = None


    def dbcv_score(self, estimator, X):
        '''
        Custom scorer function using HDBSCAN's built-in relative_validity_ metric.
        '''
        labels = estimator.fit_predict(X)
        return estimator.relative_validity_
        
    
    def determine_params(self):
        '''
        Runs GridSearchCV and returns the best-performing K-means instance.
        '''
        if self.algorithm == 'kmeans':
            param_grid = {
                'n_clusters': [3, 5, 7, 9, 11]
            }
            kmeans = KMeans(random_state=666)
            grid_search = GridSearchCV(kmeans, param_grid, cv=None, scoring=make_scorer(self.scorer))
            grid_search.fit(self.embeddings)
            return grid_search.best_estimator_
        elif self.algorithm == 'hdbscan':
            param_grid = {
                'min_cluster_size': [3, 5, 7, 9, 11, 13, 15],
                'min_samples': [2, 4, 6, 8, 10, 12],
                'cluster_selection_method': ['eom', 'leaf']
            }
            hdbscan = HDBSCAN()
            grid_search = GridSearchCV(hdbscan, param_grid, scoring=make_scorer(self.dbcv_score, greater_is_better=True, cv=None))
            grid_search.fit(self.embeddings)
            return grid_search.best_estimator_
        else:
            print(f'{self.algorithm} is not a valid clustering algorithm. Not generating clusters.')

    
    def format_result(self, cluster_labels):
        '''
        Uses the computed cluster_labels to format the result.
        The result is stored as a class attribute.
        '''
        result = {}
        unique_labels = list(collections.Counter(cluster_labels.tolist()).keys())
        for label in unique_labels:
            docs = []
            for i, v in enumerate(self.documents):
                if int(cluster_labels[i]) == label:
                    docs.append(v)
            result[label] = docs
        self.result = result

    
    def generate_clusters(self):
        '''
        Generates clusters using sklearn
        '''
        if self.algorithm == 'kmeans':
            if self.auto_score:
                kmeans = self.determine_params()
                print(f'GridSearchCV determined the best params: {kmeans}')
                cluster_labels = kmeans.predict(self.embeddings)
                self.format_result(cluster_labels)
            else:
                if self.n_clusters:
                    kmeans = KMeans(n_clusters=self.n_clusters, random_state=666)
                    cluster_labels = kmeans.fit_predict(self.embeddings)
                    self.format_result(cluster_labels)
                else:
                    print('n_clusters must be set unless auto_score is set to True')
        elif self.algorithm == 'hdbscan':
            if self.auto_score:
                hdbscan = self.determine_params()
                print(f'GridSearchCV determined the best params: {hdbscan}')
                cluster_labels = hdbscan.fit_predict(self.embeddings)
                self.format_result(cluster_labels)
            else:
                if self.n_clusters and self.n_samples:
                    hdbscan = HDBSCAN(min_cluster_size=self.n_clusters, min_samples=self.n_samples)
                    cluster_labels = hdbscan.fit_predict(self.embeddings)
                    self.format_result(cluster_labels)
                else:
                    print('n_clusters and n_samples must be set for HDBSCAN unless auto_score is set to True')
        else:
            print(f'{self.algorithm} is not a valid clustering algorithm. Not generating clusters.')