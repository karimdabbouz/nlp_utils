This is a package I wrote to do different kinds of NLP tasks in German without having to include the whole code in my Jupyter notebooks. It has modules for topic modeling, knowledge graphs, loading and cleaning data from a Postgres db as well as for creating different kinds of embeddings and token representations. 

# Modules

There are currently four different utils each including one or more modules.

## embeddings.py

Allows you to create create different kinds of representations from one or more documents. The module TransformerEmbeddings creates embeddings using Huggingface's transformers library. I will add more later...

## topic_modeling.py

Allows you to create topics and clusters from a set of documents using different techniques.

### Bertopic

The Bertopic module uses Bertopic to turn a list of documents into topics with pre-defined settings I often use.

### SklearnClustering

This module creates clusters from transformer embeddings using either K-means or HDBSCAN.

### LDA

This module creates clusters from a list of documents using LDA.

## knowledge-graph.py

This util allows you to create different knowledge graphs including their corresponding visualization based on tokens created with Stanza.

## loading_cleaning.py

This util contains a module that is highly customized to a Postgres db with news articles that I maintain for different NLP tasks. An ArticleLoader represents a connection to a DB table and has methods to load and clean data as well as basic stats.
