from transformers import AutoTokenizer, AutoModel
import torch



class TransformerEmbeddings:
    '''
    This callable class creates embeddings from a list of documents using the HuggingFace transformers library.
    Embeddings are created as an np.array of shape (num_docs, num_dimensions) and returned by calling the class.
    :param str model: The model to use
    :param str pooling_method: The pooling method
    '''
    def __init__(self, model, pooling_method):
        self.model = model
        self.pooling_method = pooling_method
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModel.from_pretrained(self.model)


    def __call__(self, documents):
        inputs = self.tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.model(**inputs)
        if self.pooling_method == 'cls':
            return output.last_hidden_state[:, 0, :].numpy()
        if self.pooling_method == 'mean':
            return output.last_hidden_state.mean(dim=1).numpy()