import yake
from yake.highlight import TextHighlighter
from torch import cuda
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
stopwords = ["anh","bị","bởi","cho","chính","chú_thích","các","còn","có","có_thể","cùng","cũng","của","do","dân_số","gọi","hai","hoa","hơn","họ","introduction","khi",
"khoa_học","khác","không","liên_kết","loài","là","làm","lại","mô_tả","một","ngoài","ngày","người","nhiều","nhà","như","nhưng","nhất","những","này","nó",
"năm","nước","of","ra","sau","sống","sự","tham_khảo","theo","thuộc","tháng","thêm","thể_loại","thứ","thực_vật","trong","trên","trước","tên","tại","tỉnh","từ","và","vào",
"vùng","về","với","xem","đây","đã","đó","được","đầu_tiên","đến","để"]

def extract_word(text):
    kw_extractor = yake.KeywordExtractor(lan='vi', n=2, stopwords=stopwords)
    keywords = kw_extractor.extract_keywords(text)
    txt = ""
    for kw in keywords:
        txt  = txt + '{:40}'.format(("<span class='textt' >" + str(kw[0]) + "</span>")) + "  " + str(kw[1]) + " <br> "
    return txt

def extract_text(text):
    kw_extractor = yake.KeywordExtractor(lan='vi', n=2, stopwords=stopwords)
    keywords = kw_extractor.extract_keywords(text)
    th = TextHighlighter(max_ngram_size = 3, highlight_pre = "<span class='textt' >", highlight_post= "</span>")
    rs = th.highlight(text, keywords)
    return rs

def process(sentence):
    device = 'cuda' if cuda.is_available() else 'cpu'
    PATH = r'C:\Users\ADMIN\Desktop\NLP_FINAL\ChuyenDeCNTTT\map\NLP\kha-vn-bert-ner'
    PATH1 = r'C:\Users\ADMIN\Desktop\NLP_FINAL\ChuyenDeCNTTT\map\NLP\kha-vn-bert-token'
    model_test = BertForTokenClassification.from_pretrained(PATH, local_files_only=True)
    tokenizer_test = BertTokenizer.from_pretrained(PATH1, local_files_only=True)
    model_test.to(device)
    id2label = {0: 'O',
                1: 'B-PER',
                2: 'B-LOC',
                3: 'I-LOC',
                4: 'I-PER',
                5: 'B-ORG',
                6: 'B-MISC',
                7: 'I-MISC',
                8: 'I-ORG'}
    inputs = tokenizer_test(sentence, padding='max_length', truncation=True, max_length=300, return_tensors="pt")
    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model_test(ids, mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model_test.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer_test.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [id2label[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

    word_level_predictions = []
    for pair in wp_preds:
        if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
        # skip prediction
            continue
        else:
         word_level_predictions.append(pair[1])

    # we join tokens, if they are not special ones
    str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']])
    tmp = ""
    count = 0
    for word in str_rep.split(" "):
        if word.startswith("##"):
            tmp = tmp.strip()
            tmp = tmp + word.replace("##","") + " "
            del word_level_predictions[count]
            continue
        tmp += word + " "
        count+=1


    return word_level_predictions

def getFinalString(text):
    sentences = text.split(".")
    rs = []
    for sentence in sentences:
        word_level_predictions = process(sentence)
        sentence = sentence.replace(',', " , ")
        sentence = sentence.replace('.', " . ")
        sentence= sentence.replace('?', " ? ")
        sentence = sentence.replace('!', " ! ")
        sentence = re.sub(' +', ' ',sentence)
        sentences_AS_list = sentence.split(" ")
        count = 0
        for ner in word_level_predictions:
            if(ner != 'O'):
                if(sentences_AS_list[count] not in [",", ".", "?", "!", "(", ")", '"', "'", "\\", "//", "{", "}"]):
                    sentences_AS_list[count] = "<span class='textt' >" + sentences_AS_list[count] + "</span>"
            count +=1

        finalRs = " ".join(sentences_AS_list)
        rs.append(finalRs)

    Final_string = ".".join(rs)
    return Final_string.strip()

