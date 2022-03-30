from config import *
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

with_global = []
without_global = []

cfg = GENIAConfig()

with open('test_results/' + cfg.dataset + '/test_result.json', 'r', encoding='utf-8') as f:
    tmp = f.read().split('}')
    for idx, line in enumerate(tmp):
        if line == '\n':
            continue
        with_global.append(json.loads(line + '}'))

with open('test_results/' + cfg.dataset + '/test_result-without-global.json', 'r', encoding='utf-8') as f:
    tmp = f.read().split('}')
    for line in tmp:
        if line == '\n':
            continue
        without_global.append(json.loads(line + '}'))

with open(cfg.ent2id_path, 'r', encoding='utf-8') as f:
    ent2id = json.load(f)
    for key, value in ent2id.items():
        ent2id[key] = np.around(value / 7, 2)

def find_head_idx(source, target):
    target = target.split()
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            yield i

def create_table(data):
    text = data['text'].split()
    pred_list = data['pred_list']
    gold_list = data['Gold_list']
    table = np.ones((len(text), len(text)))
    gold_table = np.ones((len(text), len(text)))
    for item in pred_list:
        if item in gold_list:
            type_num = 0.2
        else:
            type_num = 0.5
        target = item[0]
        length = len(target.split())-1
        target_type = item[1]
        count = 0
        for ner_begin in find_head_idx(text, target):
            count += 1
            table[ner_begin][ner_begin+length] = type_num
        if count == 0:
            raise RuntimeError('not found word')

    for item in gold_list:
        target = item[0]
        length = len(target.split())-1
        target_type = item[1]
        count = 0
        for ner_begin in find_head_idx(text, target):
            count += 1
            gold_table[ner_begin][ner_begin+length] = 0.2
        if count == 0:
            raise RuntimeError('not found word')
    return table, gold_table, text

def create_pic():
    fig, ax = plt.subplots(1, 3, figsize=(12,6))
    count = 0
    for idx, (a, b) in enumerate(zip(with_global, without_global)):
        tablea, gold_table, text = create_table(a)
        tableb, _, _ = create_table(b)

        if sum(sum(tablea != tableb)):
            count += 1
            # if count < 245:
            #     continue
            # if count > 245:
            #     break
            # start = 25
            # end = 35
            # tablea = tablea[start:end, start:end]
            # tableb = tableb[start:end, start:end]
            # gold_table = gold_table[start:end, start:end]
            # text = text[start:end]
            tablea = pd.DataFrame(tablea, index=text, columns=text)
            tableb = pd.DataFrame(tableb, index=text, columns=text)
            gold_table = pd.DataFrame(gold_table, index=text, columns=text)
            im = sns.heatmap(tablea, cbar=False, ax=ax[2], vmin=0, vmax=1, square=True, linewidth=.5, yticklabels=False)
            im1 = sns.heatmap(tableb, cbar=False, ax=ax[1], vmin=0, vmax=1, square=True, linewidth=.5, yticklabels=False)
            im2 = sns.heatmap(gold_table, cbar=False, ax=ax[0], vmin=0, vmax=1, square=True, linewidth=.5)
            ax[0].set_title('Gold Table')
            ax[1].set_title('Primary Table')
            ax[2].set_title('Prediction Table')
            # im.set_xticklabels(im.get_xticklabels(), rotation=75, horizontalalignment='right')
            # im1.set_xticklabels(im1.get_xticklabels(), rotation=75, horizontalalignment='right')
            plt.tight_layout()
            plt.savefig('figures/table_fig%d' % count)
            # plt.show()


def get_statistics():
    pred_locate_error = 0
    pred_type_error = 0
    pred_total = 0
    primary_locate_error = 0
    primary_type_error = 0
    primary_total = 0
    for idx, (a, b) in enumerate(zip(with_global, without_global)):
        tablea, gold_table, text = create_table(a)
        tableb, _, _ = create_table(b)

        for i in range(len(tablea)):
            for j in range(len(tablea[0])):
                if tablea[i][j] != 1:
                    pred_total += 1
                if tablea[i][j] == 0.5 and gold_table[i][j] != 1:
                    pred_type_error += 1
                if tablea[i][j] == 0.5 and gold_table[i][j] == 1:
                    pred_locate_error += 1

        for i in range(len(tableb)):
            for j in range(len(tableb[0])):
                if tableb[i][j] != 1:
                    primary_total += 1
                if tableb[i][j] == 0.5 and gold_table[i][j] != 1:
                    primary_type_error += 1
                if tableb[i][j] == 0.5 and gold_table[i][j] == 1:
                    primary_locate_error += 1

    print('pred total: %d\n pred type error num: %d\n pred locate error num %d \n'
          'primary total %d\n primary type error num %d\n primary locate error num %d\n'
          'pred type error rate %.4f\n pred locate error rate %.4f\n'
          'primary type error rate %.4f\n primary locate error rate %.4f' % (pred_total, pred_type_error, pred_locate_error,
          primary_total, primary_type_error, primary_locate_error, pred_type_error/pred_total, pred_locate_error/pred_total,
            primary_type_error/primary_total, primary_locate_error/primary_total))


# pred_loc = []
# primary_loc = []
#
# with open('test_results/' + cfg.dataset + '/loc_result.json', 'r', encoding='utf-8') as f:
#     for idx, line in tqdm(enumerate(f)):
#         data = json.loads(line)
#         pred_loc.append(data)
#
# with open('test_results/' + cfg.dataset + '/loc_result.json', 'r', encoding='utf-8') as f:
#     for idx, line in tqdm(enumerate(f)):
#         data = json.loads(line)
#         primary_loc.append(data)
#
# fig, ax = plt.subplots(3, 1, figsize=(6, 12))
# for idx, (dataa, datab) in enumerate(zip(primary_loc, pred_loc)):
#
#     for i in range(len(dataa['locs'])):
#         for j in range(len(dataa['locs'][0])):
#             if dataa['locs'][i][j] == 0:
#                 dataa['locs'][i][j] = 10
#     tablea = pd.DataFrame(dataa['locs'], index=dataa['tokens'], columns=dataa['tokens'])
#     for i in range(len(dataa['locs'])):
#         for j in range(len(dataa['locs'][0])):
#             if dataa['locs'][i][j] == 10:
#                 dataa['locs'][i][j] = ''
#             else:
#                 dataa['locs'][i][j] = str(dataa['locs'][i][j])
#
#     for i in range(len(datab['locs'])):
#         for j in range(len(datab['locs'][0])):
#             if datab['locs'][i][j] == 0:
#                 datab['locs'][i][j] = 10
#     tableb = pd.DataFrame(datab['locs'], index=datab['tokens'], columns=datab['tokens'])
#     for i in range(len(datab['locs'])):
#         for j in range(len(datab['locs'][0])):
#             if datab['locs'][i][j] == 10:
#                 datab['locs'][i][j] = ''
#             else:
#                 datab['locs'][i][j] = str(datab['locs'][i][j])
#
#     im = sns.heatmap(tablea, cbar=False, ax=ax[0], vmin=0, vmax=10, square=True, linewidth=.5, annot=dataa['locs'], fmt='', xlabeltick=False)
#     im1 = sns.heatmap(tableb, cbar=False, ax=ax[1], vmin=0, vmax=10, square=True, linewidth=.5, annot=datab['locs'], fmt='',
#                      xlabeltick=False)
#
#
#
#     plt.savefig('figures/figure_loc%d' % idx)

# for line in with_global:
#     if 20 <= len(line['text'].split()) <= 30 and line['pred_list'] != line['Gold_list']:
#         gold = line['Gold_list']
#         for i in range(len(gold)):
#             for j in range(len(gold)):
#                 if gold[i][0] in gold[j][0] and len(gold[i][0].split()) != len(gold[j][0].split()):
#                     print(line)
#                     break
if __name__ == '__main__':
    create_pic()