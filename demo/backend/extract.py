'''
Extraction Module
'''

import os
from os import environ
from dotenv import load_dotenv
import re
from spacy.lang.en import English
from chemrxnextractor import RxnExtractor
import warnings
from chemdataextractor.doc import Paragraph
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys

warnings.filterwarnings("ignore")

nlp = English()
tokenizer = nlp.tokenizer
extractor = None


def tokenize(text):
    text = ' '.join([t.text for t in tokenizer(text)])
    text = re.sub('\n+', '\n', text)
    text = re.sub('[ ]{2,}', ' ', text)
    text = '\n'.join([s.strip() for s in text.split('\n') if s.strip()])
    return text


def parse_xml(raw):
    soup = BeautifulSoup(raw, features="html.parser")

    doc_struct = {
        'paragraphs': [],
    }

    ## File Description Data ##
    doc_desc = soup.find('filedesc')

    title = ''
    for t in doc_desc.find_all('title'):
        title += t.text + ' '
    title = tokenize(title.strip())
    doc_struct['title'] = title

    raw_desc = ''
    for c in doc_desc.find_all():
        if c.text:
            raw_desc += ' ' + c.text.strip()
    raw_desc = tokenize(raw_desc)

    # Join all < 3 word paragraphs
    desc_paragraphs = [None]
    for p in raw_desc.split('\n'):
        if len(p.split(' ')) <= 3:
            if desc_paragraphs[-1]:
                desc_paragraphs[-1] += ' ' + p
            else:
                desc_paragraphs[-1] = p
        else:
            if desc_paragraphs[-1]:
                desc_paragraphs.append(p)
            else:
                desc_paragraphs[-1] = p
                desc_paragraphs.append(None)
    if not desc_paragraphs[-1]:
        desc_paragraphs.pop()

    # Add to data
    for p in desc_paragraphs:
        doc_struct['paragraphs'].append({
            'text': p,
            'head': 'DOCUMENT DESCRIPTION',
            'position': 'DOCUMENT DESCRIPTION',
        })

    ## Abstract ##
    abstract = soup.find('abstract')
    if abstract:
        abstract_raw = ''
        for p in abstract.find_all():
            abstract_raw += ' ' + p.text
        abstract_raw = tokenize(abstract_raw)
        doc_struct['paragraphs'].append({
            'text': abstract_raw,
            'head': 'ABSTRACT',
            'position': 'ABSTRACT',
        })

    ## Main Body ##
    position = 0
    body = soup.find('text').find('body')
    for div in body.find_all('div', recursive=False):
        head = ''
        for h in div.find_all('head'):
            head += ' ' + h.text

        for p in div.find_all('p'):
            ptext = p.text
            doc_struct['paragraphs'].append({
                'text': tokenize(ptext),
                'head': head.strip(),
                'position': f'Paragraph {position}',
            })
            position += 1

    df = pd.DataFrame(doc_struct['paragraphs'])
    df['title'] = title
    
    return df

def get_pdf_text(file):
    config_path = os.path.join(os.path.realpath('.'), '.env')
    load_dotenv(dotenv_path=config_path)

    grobid_api = environ.get('GROBID_SERVER')
    endpoint_api = f'{grobid_api}/processFulltextDocument'

    r = requests.post(endpoint_api, files={'input': file})
    df = parse_xml(r.text)

    # Paragraph Filters
    target_keywords = ['abstract','scheme','schemes','table','tables','entry','entries','report','synthesis', 'synthesized','gave','gives','cathalized','yield','yields','obtained']
    headers = ['table','scheme','figure']
    par_min_len = 25
    par_max_len = 300

    has_target_kw = df['text'].apply(lambda t: any([kw in t for kw in target_keywords]))
    has_target_len = df['text'].apply(lambda t: par_min_len <= len(t.split()) <= par_max_len)

    is_experimental_section = df['text'].apply(lambda t: 'experimental section' in t.lower()) & \
        df['head'].apply(lambda h: 'experimental section' in h.lower())

    section_blacklist = df['head'].apply(lambda t: any([h.lower() in t.lower() for h in headers]))
    include_locs = has_target_kw & has_target_len & (~is_experimental_section) & (~section_blacklist)

    df = df.loc[include_locs].copy().reset_index(drop=True)
    paragraphs = list(df['text'].values)
    title = df.iloc[0]['title']

    return title, paragraphs

def extract_documents(files):

    res = [get_pdf_text(f) for f in files]
    titles, paragraphs = zip(*res)

    # extractions = get_ents(paragraphs)
    extractions = []
    for j, p in enumerate(paragraphs):
        extractions.append({'title': titles[j], 'paragraphs': get_ents(p)})

    return extractions

def get_ents(paragraphs):

    # get extractor
    global extractor

    config_path = os.path.join(os.path.realpath('.'), '.env')
    load_dotenv(dotenv_path=config_path)
    models_dir = environ.get('MODELS_DIR')
    model_name = environ.get('ACTIVE_MODEL')
    model_dir = os.path.join(models_dir, model_name)
    
    if extractor is None:
        extractor = RxnExtractor(model_dir=model_dir)

    # Get sentences
    paragraphs = [Paragraph(p).sentences for p in paragraphs]
    sentences = []

    for par in paragraphs:
        for sent in par:
            sentences.append(str(sent))

    reactions = extractor.get_reactions(sentences)

    # Re-combine sentences into paragraphs
    extractions = []
    off = 0
    for par in paragraphs:
        tokens = []
        recs = []
        for j in range(off, off + len(par)):
            sent_react = reactions[j]
            for r in sent_react["reactions"]:
                r_offset = {}
                for k in r:
                    r_offset[k] = []
                    for e in r[k]:
                        if isinstance(e, (list, tuple)):
                            r_offset[k].append([j + len(tokens) for j in e[1:]])
                        else:
                            if isinstance(e, int):
                                r_offset[k].append(e + len(tokens))

                recs.append(r_offset)

            tokens.extend(sent_react['tokens'])

        extractions.append({'tokens': tokens, 'reactions': recs})
        off += len(par)

    return extractions


if __name__ == '__main__':
    pdf_path = sys.argv[1]

    with open(pdf_path, 'rb') as f:
        extract_documents([f])
