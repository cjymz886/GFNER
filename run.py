from transformers import WEIGHTS_NAME,AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch
from lr_scheduler import ReduceLROnPlateau
from model import NerModel, NerRobertaModel
from util import *
from config import *
from loader import *
from tqdm import tqdm
import os
import sys
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertConfig
# from transformers.models.roberta.modeling_roberta import RobertaConfig
import json


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(load_method):
    if not os.path.exists(os.path.dirname(cfg.save_path)):
        os.mkdir(os.path.dirname(cfg.save_path))
    train_data = load_method([cfg.train_path, cfg.dev_path])
    dev_data= load_method([cfg.test_path])

    if cfg.use_bert:
        if 'roberta' in cfg.bert_model:
            collator = RobertaCollator(cfg,tokenizer,entity2id)
        else:
            collator=Collator(cfg,tokenizer,entity2id)
    else:
        collator = LSTMCollator(cfg, tokenizer, entity2id, token2id)

    data_loader=DataLoader(train_data,collate_fn=collator,batch_size=cfg.batch_size,num_workers=0)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.min_num)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)

    best_f1 = -1.0
    step = 0
    crossentropy=nn.CrossEntropyLoss(reduction="none")
    # SmoothLoss= LabelSmoothLoss(smoothing=0.01)
    # focalloss=focal_loss(alpha=0.25, gamma=2, num_classes = len(entity2id))

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=data_loader.__len__(), desc="train", ncols=80) as t:
            for i, batch in enumerate(data_loader):
                batch = [d.to(cfg.device) for d in batch]
                batch_token, batch_token_mask,batch_label,batch_label_mask= batch
                table = model(batch_token, batch_token_mask) # BLLY
                table=table.reshape([-1,len(entity2id)])
                batch_label=batch_label.reshape([-1])

                #CE
                loss=crossentropy(table,batch_label.long())
                loss=(loss*batch_label_mask.reshape([-1])).sum()

                #CE_smoothing
                # loss=SmoothLoss(table,batch_label.long())
                # loss = (loss * batch_label_mask.reshape([-1])).sum()

                #focal loss
                # loss=focalloss(table,batch_label.long())
                # loss = (loss * batch_label_mask.reshape([-1])).sum()

                loss.backward()
                step += 1
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                model.zero_grad()
                t.set_postfix(loss="%.4lf"%(loss.cpu().item()))
                t.update(1)

        print("")
        precision, recall, f1_score = evaluate(model, cfg, dev_data,tokenizer,id2entity,output_path=None,is_test=False)
        logs={'precision':precision,'recall':recall,'f1_score':f1_score}
        show_info = f'Epoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        print(show_info)
        scheduler.epoch_step(logs['f1_score'], epoch)
        if logs['f1_score'] > best_f1:
            best_f1= logs['f1_score']
            if isinstance(model, nn.DataParallel):
                model_stat_dict = model.module.state_dict()
            else:
                model_stat_dict = model.state_dict()
            state = {'epoch': epoch, 'state_dict': model_stat_dict}
            torch.save(state, cfg.save_path)


def test(load_method):
    test_data = load_method([cfg.test_path])

    states = torch.load(cfg.save_path)
    state = states['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)


    test_result_path = 'test_results/' + cfg.dataset + '/test_result.json'
    if not os.path.exists('test_results/' + cfg.dataset):
        os.mkdir('test_results/' + cfg.dataset)
    precision, recall, f1_score = evaluate(model, cfg, test_data,tokenizer,id2entity,output_path=test_result_path, is_test=True)
    print(f'{precision}\t{recall}\t{f1_score}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='TFNER')
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    parser.add_argument('--config_name', default='ace', type=str, help='ace or genia or weibo or conll')
    args = parser.parse_args()

    cfg_dict = {
        'ace': Ace2005Config(),
        'genia': GENIAConfig(),
        'conll': ConllConfig(),
        'weibo': WeiboConfig()
    }

    cfg = cfg_dict[args.config_name]

    entity2id=json.load(open(cfg.ent2id_path))
    id2entity=dict(zip(entity2id.values(), entity2id.keys()))

    #########################
    token2id = json.load(open(cfg.token2id_path))
    #########################
    from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
    from transformers.models.roberta.modeling_roberta import RobertaConfig
    if 'roberta' in cfg.bert_model:
        tokenizer = RobertaTokenizer.from_pretrained(cfg.bert_path)
        config = RobertaConfig.from_pretrained(cfg.bert_path)
    else:
        tokenizer = get_tokenizer(cfg.bert_vocab_path, is_eng=cfg.is_eng)
        config = BertConfig.from_pretrained(cfg.bert_config_path)

    config.num_enity=len(id2entity)
    config.fix_bert_embeddings=cfg.fix_bert_embeddings
    config.token2id = token2id
    config.word2vector = cfg.word2vector
    config.use_bert = cfg.use_bert
    config.vocab_s = cfg.vocab_size
    config.pred_mode = cfg.pred_mode
    config.bert_path = cfg.bert_path
    if 'roberta' in cfg.bert_model:
        model = NerRobertaModel(config)
    else:
        model=NerModel.from_pretrained(pretrained_model_name_or_path=cfg.bert_checkpoint_path, config=config)
    model.to(cfg.device)

    load_methods = {
            'ace': load_ace,
            'genia': new_load_genia,
            'weibo': load_weibo,
            'conll': load_conll
        }

    if args.mode == 'train':
        train(load_methods[args.config_name])
    else:
        test(load_methods[args.config_name])
