import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

import pymorphy2

sw = stopwords.words('russian')

def load_morph():  # загрузка MorphAnalyzer
    morph = pymorphy2.MorphAnalyzer()
    return morph

punctuation = '[\'\"!"#$%&()*\+,-\./:;<=>?@\[\]^_`{|}~„“«»*\—/\-‘’…–\.+☺❤๏̯͡๏ツتبالك\_]'


# функция обработки текста

def preprocessing(text, morph):
    text = text.strip()
    clean_words = [re.sub('[A-Za-z]', '', w) for w in word_tokenize(text)]
    clean_words[0] = re.sub('\\ufeff', '', clean_words[0])  # удаление этого тэга из начала файлов
    clean_words = [re.sub('[0-9]', '', w) for w in clean_words]
    clean_words = [w.strip(punctuation) for w in clean_words]
    clean_words = [re.sub('[\.-\/-\-]', ' ', w) for w in clean_words]
    clean_words = [re.sub('о_о', '', w) for w in clean_words]
    clean_words = [w.lower() for w in clean_words if w != '']
    clean_words = [w for w in clean_words if w not in sw]
    clean_words = [morph.parse(w)[0].normal_form for w in clean_words if w]
    clean_words = [w for w in clean_words if w not in sw]

    clean_words = str(clean_words)
    clean_words = re.sub('[,\[\]\']', '', clean_words)

    return clean_words


# получение словаря - полностью предобработанного корпуса

def corp_preprocessed(texts):
    corp_prep = []
    morph = load_morph()
    for i in range(len(texts)):
        text_i = preprocessing(text = texts[i], morph = morph)
        corp_prep.append(text_i)

    return corp_prep


# функция предобработки запроса

def query_preprocessed(query):
    morph = load_morph()
    query_prep = preprocessing(text = query, morph = morph)

    return query_prep