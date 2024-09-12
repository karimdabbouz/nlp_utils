from transformers import AutoTokenizer, AutoModel
import torch



class TransformerEmbeddings():
    '''
    This class creates embeddings from a list of documents using the HuggingFace transformers library.
    Embeddings are created at instantiation as an np.array of shape (num_docs, num_dimensions) and stored in the embeddings attribute.
    :param str model: The model to use
    :param str pooling_method: The pooling method
    :param list[str] documents: The list of documents to embed
    '''
    def __init__(self, model, pooling_method, documents):
        self.model = model
        self.pooling_method = pooling_method
        self.documents = documents
        self.embeddings = self.generate_embeddings()


    def generate_embeddings(self):
        '''
        Creates embeddings and stores in the class attribute.
        Runs at instantiation.
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model = AutoModel.from_pretrained(self.model)
        inputs = tokenizer(self.documents, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = model(**inputs)
        if self.pooling_method == 'cls':
            return output.last_hidden_state[:, 0, :].numpy()
        if self.pooling_method == 'mean':
            return output.last_hidden_state.mean(dim=1).numpy()