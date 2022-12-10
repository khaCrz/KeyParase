from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Search
from .forms import SearchForm
from django.http import JsonResponse

from smart_open import smart_open
import json
from . import data_prep
from . import yake_process

# Create your views here.
class WikiParser:
    def __init__(self, wiki_json_dump_file):
        self.wiki_json_dump_file = wiki_json_dump_file
        self.tp = data_prep.TextPreprocess()

    def parse_txt(self):
        count = 0
        doc = []
        for line in smart_open(self.wiki_json_dump_file):
            article = json.loads(line.decode('utf8'))
            # each article has a "title",
            # a mapping of interlinks and a list of "section_titles" and "section_texts".
            texts = [article['title']]
            for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                texts.append(section_title)
                texts.append(section_text)
            article_text = ' '.join(texts)
            doc.append(Docc(texts[0], article_text[len(texts[0]) + 13:]))
            count+=1
            if(count >= 10): break
        return doc

    def predata(self):
        count = 0
        doc = []
        for line in smart_open(self.wiki_json_dump_file):
            article = json.loads(line.decode('utf8'))
            # each article has a "title",
            # a mapping of interlinks and a list of "section_titles" and "section_texts".
            texts = [article['title']]
            for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                texts.append(section_title)
                texts.append(section_text)
            article_text = self.tp.preprocess(' '.join(texts), tokenize=True)
            doc.append(Docc(texts[0], article_text[len(texts[0]) + 13:]))
            count+=1
            if(count >= 10): break
        return doc

class Docc:
    def __init__(self, title, doc):
        self.title = title
        self.doc = doc

def index(request):
    return render(request, 'index.html')

def preview(request):
    wiki_parser = WikiParser(r'C:\Users\ADMIN\Downloads\viwiki-latest-pages-articles.json.gz').parse_txt()
    context = {
        'doc': wiki_parser
    }
    return render(request, 'Preview.html', context)

def predata(request):
    wiki_parser = WikiParser(r'C:\Users\ADMIN\Downloads\viwiki-latest-pages-articles.json.gz').predata()
    context = {
        'doc': wiki_parser
    }
    return render(request, 'PreData.html', context)

def extract(request):
    result = request.GET.get('result', None)
    text = yake_process.getFinalString(result)
    # Any process that you want
    data = {
        'text' : text
    }
    return JsonResponse(data)
