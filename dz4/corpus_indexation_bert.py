# код с гугл коллаба для получения и скачивания индексированного бертом корпуса

from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru").cuda()

a = 0
b = 500

berted_corp = []

for i in range(100):

    #Tokenize sentences
    encoded_input = tokenizer(corpus_preprocessed[a:b], padding=True, truncation=True, max_length=24, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input.to('cuda'))

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    berted_corp.extend(sentence_embeddings)

    a += 500
    b += 500


berted_corp_path = "C:\\Users\marga\Downloads\\berted_corp.pt"
torch.save(berted_corp, berted_corp_path)