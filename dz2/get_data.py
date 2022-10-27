import os

# здесь сначала создается словарь путей до файлов, затем читаются сами файлы внутри корпуса
# потом создается словарь названий файлов (можно номеров/индексов, но у нас есть названия)

def get_texts(f_path):
    f_paths = []
    for root, dirs, files in os.walk(f_path):
        for name in files:
            if ".DS_Store" not in name:
                f_paths.append(os.path.join(root, name))

    texts = []
    for i in range(len(f_paths)):
        with open(f_paths[i], 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)

    f_names = []
    for i in range(len(f_paths)):
        f_names.append(f_path[i])  # названиями будем считать пути до файлов

    return texts, f_names, f_paths