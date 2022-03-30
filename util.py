from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import codecs
from bert4keras.tokenizers import Tokenizer



class NERTokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._do_lower_case:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
            tokens.append('[unused1]')
        return tokens

def get_tokenizer(vocab_path,is_eng=True):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    if is_eng:
        return NERTokenizer(token_dict, do_lower_case=True)
    else:
        return Tokenizer(token_dict, do_lower_case=True)


def one_hot(x, class_num):
    return torch.eye(class_num)[x,:]




class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, reduction="none"):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss,self).__init__()
        self.reduction = reduction
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())

        if self.reduction=='none':
            return loss
        elif self.reduction=="mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss




class LabelSmoothLoss(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1)
        return loss






def extract_entity(model,cfg,tokenizer,text,id2entity):
    if cfg.use_bert:
        if 'roberta' in cfg.bert_model:
            tokens_id = tokenizer.encode(text)
            tokens = tokenizer.tokenize(text)
        else:
            tokens_id,_= tokenizer.encode(text, maxlen=cfg.seq_len)
            tokens = tokenizer.tokenize(text, maxlen=cfg.seq_len)
    else:
        token2id = json.load(open(cfg.token2id_path))
        tokens_id = [token2id[x] for x in text]
        tokens = text
    mask = [1 if t > 0 else 0 for t in tokens_id]

    tokens_id=torch.LongTensor([tokens_id]).to(cfg.device)
    mask=torch.Tensor([mask]).to(cfg.device)

    model.eval()

    with torch.no_grad():
        table=model(tokens_id, mask) #BLLR
        table = table.cpu().detach().numpy() #BLLY
        all_loc = table[0].argmax(axis=-1)

        res=[]
        for i in range(len(all_loc[0])):
            for j in range(i,len(all_loc[0])):
                type_idx=all_loc[i,j]
                if type_idx!=0:
                    type=id2entity[type_idx]

                    if cfg.is_eng:

                        if cfg.use_bert:
                            if 'roberta' in cfg.bert_model:
                                ent = tokens[i:j]
                                ent = ''.join(ent).split('Ġ')
                                ent = ' '.join(ent).strip()
                            else:
                                ent = tokens[i:j]
                                ent = ''.join([i.lstrip("##") for i in ent])
                                ent = ' '.join(ent.split('[unused1]'))
                        else:
                            ent = tokens[i:j + 1]
                            ent = ' '.join(ent)
                        if len(ent) >= 1:
                            res.append([ent, type])
                    else:
                        ent=tokens[i:j+1]
                        ent=''.join(ent)
                        if len(ent) >= 1:
                            res.append([ent, type])

        loc_path = 'test_results/' + cfg.dataset + '/loc_result-without_global.json'
        w = {'tokens': tokens, 'locs': all_loc.tolist()}
        with open(loc_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(w, ensure_ascii=False) + '\n')
        return res





def evaluate(model, cfg, eval_data,tokenizer,id2entity,output_path=None,is_test=False):

    if output_path:
        F = open(output_path, 'w')
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10

    i=1
    for line in tqdm(iter(eval_data)):
        if cfg.is_eng:
            text = ' '.join(line.tokens)
        else:
            text = ''.join(line.tokens)

        if cfg.use_bert:
            Pred_list = extract_entity(model, cfg, tokenizer, text, id2entity)
        else:
            Pred_list = extract_entity(model,cfg,tokenizer,line.tokens,id2entity)
        Gold_list = []
        for e in line.entities:
            if cfg.is_eng:
                entity =' '.join(line.tokens[e[0]:e[1]])
            else:
                entity = ''.join(line.tokens[e[0]:e[1]])
            entity_type=e[2]
            Gold_list.append([entity,entity_type])
        #
        # print('\nPred_list:',Pred_list)
        # print('Gold_list:',Gold_list)

        if is_test:
            if not Gold_list and not Pred_list:
                correct_num+=1
                predict_num+=1
                gold_num+=1

        correct_num += len([t for t in Pred_list if t in Gold_list])
        predict_num += len(Pred_list)
        gold_num += len(Gold_list)

        if output_path:
            result = json.dumps({
                'text': text,
                'pred_list':Pred_list ,
                'Gold_list': Gold_list
            }, ensure_ascii=False, indent=4)
            F.write(result + '\n')


    if output_path:
        F.close()

    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)

    print('correct_num:',correct_num,'predict_num:',predict_num,'gold_num',gold_num)
    return precision, recall, f1_score