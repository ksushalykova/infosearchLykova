import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel


def get_bert_matrix(bert_matrix_path):   # загрузка матрицы для bert
    corpus_embeddings = torch.load(bert_matrix_path, map_location=torch.device('cpu'))
    bert_corpus_vectors = []
    for i in corpus_embeddings:
        i = i.detach().numpy()
        bert_corpus_vectors.append(i)

    bert_matrix = np.array(bert_corpus_vectors)
    return bert_matrix


def load_bert_model():  # загрузка токенизатора и модели bert
    bert_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    bert_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    return bert_tokenizer, bert_model

bert_tokenizer, bert_model = load_bert_model()  # зафиксируем модель и токенайзер

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# функция индексации query с помощью bert
def query_vec_bert_indexation(bert_tokenizer, bert_model, query_prep):
    # токенизация
    encoded_input = bert_tokenizer(query_prep, padding=True, truncation=True, max_length=24, return_tensors='pt')
    # эмбеддинги
    with torch.no_grad():
        model_output = bert_model(**encoded_input)
    # mean pooling
    q = mean_pooling(model_output, encoded_input['attention_mask'])

    query_vec_bert = q.numpy()
    return query_vec_bert