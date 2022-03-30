import codecs
from collections import namedtuple
import numpy as np
import torch
import json
import collections
import re

def load_genia(filename):
    SentInst = namedtuple('SentInst', 'tokens entities')

    sent_list = []
    max_len=0
    max_span=0
    label_set=[]
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "":  # last few blank lines
                break

            tokens = line.split(' ')
            entities = next(f).strip()
            if entities == "":  # no entities
                sent_inst = SentInst(tokens, [])
            else:
                entity_list = []
                entities = entities.split("|")
                for item in entities:
                    pointers, label = item.split()
                    pointers = pointers.split(",")
                    if int(pointers[1]) > len(tokens):
                        print('The input data maybe wrong at %s',line)

                    if len(tokens) > max_len:
                        max_len=len(tokens)

                    new_entity = (int(pointers[0]), int(pointers[1]), label)
                    entity_list.append(new_entity)
                    label_set.append(label)

                    if (int(pointers[1])-int(pointers[0]))>max_span:
                        max_span=int(pointers[1])-int(pointers[0])

                sent_inst = SentInst(tokens,entity_list)
            assert next(f).strip() == ""

            sent_list.append(sent_inst)

    # labels=list(set(label_set))
    # label2id=dict(zip(labels,range(len(labels))))

    print("Max tokens: {}".format(max_len))
    # print("Max span: {}".format(max_span))

    return sent_list


def new_load_genia(filenames):
    SentInst = namedtuple('SentInst', 'tokens entities')

    sent_list = []
    max_len = 0
    max_span = 0
    label_set = []
    for filename in filenames:
        with open(filename) as f:
            data = json.loads(f.read())
            for line in data:

                tokens = line['tokens']
                entities = []
                for item in line['entities']:
                    entities.append((item['start'], item['end'], item['type']))

                if len(tokens) > max_len:
                    max_len = len(tokens)

                sent_inst = SentInst(tokens, entities)

                sent_list.append(sent_inst)

        # labels=list(set(label_set))
        # label2id=dict(zip(labels,range(len(labels))))

        print("Max tokens: {}".format(max_len))
        # print("Max span: {}".format(max_span))

    return sent_list


def load_weibo(filenames):
    SentInst = namedtuple('SentInst', 'tokens entities')

    sent_list = []
    max_len = 0
    label_set=[]
    max_span=0
    for filename in filenames:
        with open(filename) as f:
            for sentence in f.read().split('\n\n'):
                if len(sentence)<1:
                    break
                sen_split=[s.strip().split('\t') for s in sentence.split('\n')]
                tokens=[w[0] for w in sen_split]
                entities=[]
                for i,(c1,t1) in enumerate(sen_split):
                    if 'B-' in t1:
                        label=t1.split('-')[1]
                        label_set.append(label)

                        for j,(c2,t2) in enumerate(sen_split[i+1:],i+1):
                            if 'B-' in t2:
                                entities.append((i,j,label))
                                break
                            elif 'I-' in t2:
                                if j==len(sen_split)-1:
                                    entities.append((i, j+1, label))
                                continue
                            if 'O' in t2:
                                entities.append((i,j,label))
                                break
                        if i==len(sen_split)-1:
                            entities.append((i, i+1, label))
                    else:
                        continue
                sent_inst = SentInst(tokens, entities)

                if len(entities)>0:
                    span=max([ item[1]-item[0] for item in entities])
                    if span>max_span:
                        max_span=span

                if len(tokens)>max_len:
                    max_len=len(tokens)
                sent_list.append(sent_inst)

        labels=['O']+list(set(label_set))
        label2id=dict(zip(labels,range(len(labels))))

        print("Max length: {}".format(max_len))
        print("max span: {}".format(max_span))
        print("label set: {}".format(label2id))
        print('# sentences {}'.format(len(sent_list)))
        print('avg sentences length {}'.format(np.mean([len(x.tokens) for x in sent_list])))
        print('# total entities {}'.format(np.sum([len(x.entities) for x in sent_list])))
    return sent_list


def load_conll(filenames):
    SentInst = namedtuple('SentInst', 'tokens entities')

    sent_list = []
    max_len = 0
    label_set=[]
    max_span=0
    for filename in filenames:
        with open(filename) as f:
            for sentence in f.read().split('\n\n'):
                if len(sentence)<1:
                    break

                sen_split=[s.rstrip() for s in sentence.split('\n')]
                tokens=[]

                entities=[]
                for i,items in enumerate(sen_split):
                    items=items.split(' ')
                    assert len(items)==4
                    tokens.append(items[0])
                    t1=items[-1]
                    if 'B-' in t1:
                        label=t1.split('-')[1]
                        label_set.append(label)

                        for j,items_next in enumerate(sen_split[i+1:],i+1):
                            items_next=items_next.split(' ')
                            t2 = items_next[-1]
                            if 'B-' in t2:
                                entities.append((i,j,label))
                                break
                            elif 'I-' in t2:
                                if j==len(sen_split)-1:
                                    entities.append((i, j+1, label))
                                continue
                            if 'O' in t2:
                                entities.append((i,j,label))
                                break
                        if i==len(sen_split)-1:
                            entities.append((i, i+1, label))
                    else:
                        continue
                sent_inst = SentInst(tokens, entities)

                if len(entities)>0:
                    span=max([ item[1]-item[0] for item in entities])
                    if span>max_span:
                        max_span=span

                if len(tokens)>max_len:
                    max_len=len(tokens)
                sent_list.append(sent_inst)

        labels=['O']+list(set(label_set))
        label2id=dict(zip(labels,range(len(labels))))

        print("Max length: {}".format(max_len))
        print("max span: {}".format(max_span))
        print("label set: {}".format(set(label2id)))
        print('# sentences {}'.format(len(sent_list)))
        print('avg sentences length {}'.format(np.mean([len(x.tokens) for x in sent_list])))
        print('# total entities {}'.format(np.sum([len(x.entities) for x in sent_list])))
    return sent_list


def load_ace(filenames):
    sent_list = []
    for filename in filenames:
        with codecs.open(filename,'r') as f:
            data=json.load(f)

        org_label=['PER','ORG','GPE','LOC','FAC','WEA','VEH']

        SentInst = namedtuple('SentInst', 'tokens entities')

        max_len = 0
        label_set=[]
        max_span=0

        for line in data:
            tokens=line['words']
            entities=[]
            for item in line['golden-entity-mentions']:
                label=item['entity-type'].split(':')[0]
                if label not in org_label:
                    continue
                start=item['head']['start']
                end=item['head']['end']
                entities.append((start,end,label))

                label_set.append(label)
                span = end-start
                if span > max_span:
                    max_span = span

            sent_inst = SentInst(tokens, entities)

            if len(tokens) > max_len:
                max_len = len(tokens)
            sent_list.append(sent_inst)

        labels=['O']+list(set(label_set))
        label2id=dict(zip(labels,range(len(labels))))

        print('num samples: ', len(sent_list))
        print("Max length: {}".format(max_len))
        print("max span: {}".format(max_span))
        print("label set: {}".format(set(label2id)))
    return sent_list


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            yield i


def sequence_padding(inputs,dim=0, length=None, padding=0):
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[dim] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return torch.LongTensor(outputs)

def mat_padding(inputs, length=None, padding=0):

    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[0] for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        pad_width[0] = (0, length - x.shape[0])
        pad_width[1] = (0, length - x.shape[0])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    # return np.array(outputs)
    return torch.Tensor(outputs)


class Collator(object):
    def __init__(self, cfg, tokenizer, entity2id):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.entity2id = entity2id

    def __call__(self, batch):

        batch_token, batch_token_mask = [], []
        batch_label, batch_label_mask = [], []

        for line in batch:
            text = ' '.join(line.tokens)

            tokens = self.tokenizer.tokenize(text, maxlen=self.cfg.seq_len)
            text_len = len(tokens)
            token_ids, _ = self.tokenizer.encode(text, maxlen=self.cfg.seq_len)

            assert text_len == len(token_ids)

            mask = [1 if t > 0 else 0 for t in token_ids]

            label = np.zeros([text_len, text_len])
            # label=['N','Y']
            for e in line.entities:
                entity = ' '.join(line.tokens[e[0]:e[1]])
                entity_label = e[2]
                entity_token = self.tokenizer.tokenize(entity)[1:-1]
                for ner_begin_idx in find_head_idx(tokens, entity_token):
                    ner_end_idx = ner_begin_idx + len(entity_token) - 1
                    label[ner_begin_idx,ner_end_idx] =self.entity2id[entity_label]
                    label[ner_end_idx,ner_begin_idx]=self.entity2id[entity_label]

            mask_label = np.ones(label.shape)
            for i in range(len(token_ids)):
                for j in range(len(token_ids)):
                    if i > j:
                        mask_label[i, j] = 0

            batch_token.append(token_ids)
            batch_token_mask.append(mask)
            batch_label.append(label)
            batch_label_mask.append(mask_label)

        batch_token = sequence_padding(batch_token)
        batch_token_mask = sequence_padding(batch_token_mask)
        batch_label = mat_padding(batch_label)
        batch_label_mask = mat_padding(batch_label_mask)

        return batch_token, batch_token_mask, batch_label, batch_label_mask


class RobertaCollator(object):
    def __init__(self, cfg, tokenizer, entity2id):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.entity2id = entity2id

    def find_head_idx(self, source, target):
        target_len = len(target)
        for i in range(len(source)):
            if [re.sub('Ġ', '', x) for x in source[i: i + target_len]] == [re.sub('Ġ', '', x) for x in target]:
                yield i

    def __call__(self, batch):

        batch_token, batch_token_mask = [], []
        batch_label, batch_label_mask = [], []

        for line in batch:
            text = ' '.join(line.tokens)

            tokens = self.tokenizer.tokenize(text)
            tokens = ['[SEP]'] + tokens + ['[CLS]']
            text_len = len(tokens)
            token_ids = self.tokenizer.encode(text)

            assert text_len == len(token_ids)

            mask = [1 if t > 0 else 0 for t in token_ids]

            label = np.zeros([text_len, text_len])
            # label=['N','Y']
            for e in line.entities:
                entity = ' '.join(line.tokens[e[0]:e[1]])
                entity_label = e[2]
                entity_token = self.tokenizer.tokenize(entity)
                count = 0
                for ner_begin_idx in self.find_head_idx(tokens, entity_token):
                    count += 1
                    ner_end_idx = ner_begin_idx + len(entity_token) - 1
                    label[ner_begin_idx,ner_end_idx] =self.entity2id[entity_label]
                    label[ner_end_idx,ner_begin_idx]=self.entity2id[entity_label]
                if count == 0:
                    raise RuntimeError('label not find')

            mask_label = np.ones(label.shape)
            for i in range(len(token_ids)):
                for j in range(len(token_ids)):
                    if i > j:
                        mask_label[i, j] = 0

            batch_token.append(token_ids)
            batch_token_mask.append(mask)
            batch_label.append(label)
            batch_label_mask.append(mask_label)

        batch_token = sequence_padding(batch_token)
        batch_token_mask = sequence_padding(batch_token_mask)
        batch_label = mat_padding(batch_label)
        batch_label_mask = mat_padding(batch_label_mask)

        return batch_token, batch_token_mask, batch_label, batch_label_mask


class LSTMCollator:
    def __init__(self, cfg, tokenizer, entity2id, token2id):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.token2id = token2id
        self.entity2id = entity2id

    def __call__(self, batch):

        batch_token, batch_token_mask = [], []
        batch_label, batch_label_mask = [], []

        for line in batch:
            # text = ' '.join(line.tokens)

            # tokens = self.tokenizer.tokenize(text, maxlen=self.cfg.seq_len)
            tokens = line.tokens
            text_len = len(tokens)
            token_ids = [self.token2id[x] for x in line.tokens]

            assert text_len == len(token_ids)

            mask = [1 if t > 0 else 0 for t in token_ids]

            label = np.zeros([text_len, text_len])
            # label=['N','Y']
            for e in line.entities:
                entity = ' '.join(line.tokens[e[0]:e[1]])
                entity_label = e[2]
                # entity_token = self.tokenizer.tokenize(entity)[1:-1]
                entity_token = tokens[e[0]:e[1]]
                for ner_begin_idx in find_head_idx(tokens, entity_token):
                    ner_end_idx = ner_begin_idx + len(entity_token) - 1
                    label[ner_begin_idx,ner_end_idx] =self.entity2id[entity_label]
                    label[ner_end_idx,ner_begin_idx]=self.entity2id[entity_label]

            mask_label = np.ones(label.shape)
            for i in range(len(token_ids)):
                for j in range(len(token_ids)):
                    if i > j:
                        mask_label[i, j] = 0

            batch_token.append(token_ids)
            batch_token_mask.append(mask)
            batch_label.append(label)
            batch_label_mask.append(mask_label)

        batch_token = sequence_padding(batch_token)
        batch_token_mask = sequence_padding(batch_token_mask)
        batch_label = mat_padding(batch_label)
        batch_label_mask = mat_padding(batch_label_mask)

        return batch_token, batch_token_mask, batch_label, batch_label_mask


def get_entity2id(filepaths, savepath):


    tags = {}
    i = 0
    for file in filepaths:
        with open(file, 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                tag = line.split('\t')[1].strip()
                if tag != 'O':
                    tag = tag.split('-')[1]

                if tag not in tags:
                    tags[tag] = i
                    i += 1

    with open(savepath, 'w') as f:
        f.write(json.dumps(tags, ensure_ascii=False))


def rewrite_tags(file):
    out = []
    with open(file, 'r') as f:
        for line in f:
            if line == '\n':
                out.append(line)
                continue
            tag = line.split('\t')[1].strip()
            if tag != 'O':
                out.append(line)
            else:
                out.append(line.split('\t')[0]+'\t'+'N/A'+'\n')
    with open(file, 'w') as f:
        f.write(''.join(out))


def get_token2id(sent_list):
    c = collections.Counter()
    for item in sent_list:
        tokens = item.tokens
        for t in tokens:
            c[t] += 1
    out = {}
    i = 1
    for (x, value) in c.most_common():
        out[x] = i
        i += 1
    return out

if __name__ == '__main__':
    from config import *
    # cfg = Ace2005Config()
    # cfg = GENIAConfig()
    # cfg = WeiboConfig()
    cfg = ConllConfig()
    # label2id = load_ace(cfg.train_path)
    # with open(cfg.ent2id_path, 'w') as f:
    #     f.write(json.dumps(label2id, ensure_ascii=False))
    # rewrite_tags(cfg.test_path)

    train_sent_list = load_conll(cfg.train_path)
    dev_sent_list = load_conll(cfg.dev_path)
    test_sent_list = load_conll(cfg.test_path)
    token_ids = get_token2id(train_sent_list+dev_sent_list+test_sent_list)
    with open(cfg.token2id_path, 'w') as f:
        f.write(json.dumps(token_ids, ensure_ascii=False))
