import torch

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