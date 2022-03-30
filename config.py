import torch



class GENIAConfig():
    vocab_size=21609
    embedding_size=128
    seq_len=170
    random_seed = 2009

    hidden_dropout_prob=0.1
    learning_rate=3e-5
    weight_decay=0.0
    max_grad_norm=1.0
    warmup=0.0
    fix_bert_embeddings=False
    min_num=1e-7
    use_bert=True

    batch_size=16
    epochs=50


    device = torch.device("cuda")


    dataset ='GENIA'
    is_eng=True
    train_path = './data/' + dataset + '/genia_train_dev_context.json'
    dev_path = './data/' + dataset + '/genia.dev'
    test_path='./data/'+dataset+'/genia_test_context.json'
    ent2id_path='./data/'+dataset+'/entity2id.json'
    token2id_path = './data/' + dataset + '/token2id.json'
    save_path = './save_models/'+ dataset+'/best-model-large.bin'
    history_path='./saved_models/'+dataset+'/history.pickle'
    primary_tables = './data/' + dataset + '/primary_tables.json'
    pred_tables = './data/' + dataset + '/pred_tables.json'
    gfl = True

    # bert_model = 'biobert-base-cased-v1.2'
    bert_model='biobert-large-cased-v1.1'
    bert_path = r'E:\pretraing_models\torch/' + bert_model
    bert_config_path = r'E:\pretraing_models\torch/' + bert_model + '/config.json'
    bert_vocab_path = r'E:\pretraing_models\torch/' + bert_model + '/vocab.txt'
    bert_checkpoint_path = r'E:\pretraing_models\torch/' + bert_model + '/pytorch_model.bin'
    word2vector = r'E:\pretraing_models\torch\glove.840b.cased\glove.840B.300d.txt'


class WeiboConfig():
    vocab_size=21128
    embedding_size=128
    seq_len=178
    random_seed = 2009

    hidden_dropout_prob=0.1
    learning_rate=3e-5
    weight_decay=0.0
    max_grad_norm=1.0
    warmup=0.0
    fix_bert_embeddings=False
    min_num=1e-7
    use_bert = True

    batch_size=16
    epochs=50


    device = torch.device("cuda")


    dataset ='weiboNer'
    is_eng=False
    train_path = './data/' + dataset + '/weiboNer.train'
    dev_path = './data/' + dataset + '/weiboNer.dev'
    test_path='./data/'+dataset+'/weiboNer.test'
    ent2id_path='./data/'+dataset+'/entity2id.json'
    token2id_path = './data/' + dataset + '/token2id.json'
    save_path = './save_models/'+ dataset+'/best-model.bin'
    history_path='./saved_models/'+dataset+'/history.pickle'
    primary_tables = './data/' + dataset + '/primary_tables.json'
    pred_tables = './data/' + dataset + '/pred_tables.json'
    gfl = True

    # bert_model = 'chinese_roberta_wwm_ext_pytorch'
    bert_model = 'bert-base-chinese'
    # bert_model = 'RoBERTa_zh_L12_PyTorch'
    bert_path = r'E:\pretraing_models\torch/' + bert_model
    bert_config_path = r'E:\pretraing_models\torch/' + bert_model + '/config.json'
    bert_vocab_path = r'E:\pretraing_models\torch/' + bert_model + '/vocab.txt'
    bert_checkpoint_path = r'E:\pretraing_models\torch/' + bert_model + '/pytorch_model.bin'
    word2vector = r'E:\pretraing_models\torch/glove.840b.cased/glove.840B.300d.txt'


class ConllConfig():
    vocab_size=30290
    embedding_size=128
    seq_len=127
    random_seed = 2009

    hidden_dropout_prob=0.1
    learning_rate=3e-5
    weight_decay=0.0
    max_grad_norm=1.0
    warmup=0.0
    fix_bert_embeddings=False
    min_num=1e-7
    use_bert = True

    batch_size=6
    epochs=50


    device = torch.device("cuda")


    dataset ='CoNLL2003'
    is_eng=True
    train_path = './data/' + dataset + '/train.txt'
    dev_path = './data/' + dataset + '/valid.txt'
    test_path='./data/'+dataset+'/test.txt'
    ent2id_path='./data/'+dataset+'/entity2id.json'
    token2id_path = './data/' + dataset + '/token2id.json'
    save_path = './save_models/'+ dataset+'/best-model.bin'
    history_path='./saved_models/'+dataset+'/history.pickle'
    primary_tables = './data/' + dataset + '/primary_tables.json'
    pred_tables = './data/' + dataset + '/pred_tables.json'
    gfl = True

    # bert_model = 'biobert-base-cased-v1.2'
    # bert_model='biobert-large-cased-v1.1'
    bert_model = 'bert-large-cased'
    # bert_model = 'roberta-large'

    bert_path = r'E:\pretraing_models\torch/' + bert_model
    bert_config_path = r'E:\pretraing_models\torch/' + bert_model + '/config.json'
    bert_vocab_path = r'E:\pretraing_models\torch/' + bert_model + '/vocab.txt'
    bert_checkpoint_path = r'E:\pretraing_models\torch/' + bert_model + '/pytorch_model.bin'
    word2vector = r'E:\pretraing_models\torch/glove.840b.cased/glove.840B.300d.txt'


class Ace2005Config():
    vocab_size=20196
    embedding_size=128
    seq_len=115
    random_seed = 2009

    hidden_dropout_prob=0.1
    learning_rate=3e-5
    weight_decay=0.0
    max_grad_norm=1.0
    warmup=0.0
    fix_bert_embeddings=False
    min_num=1e-7
    use_bert=True

    batch_size=8
    epochs=50
    gfl=True


    device = torch.device("cuda")


    dataset ='ACE2005'
    is_eng=True
    train_path = './data/' + dataset + '/train.json'
    dev_path = './data/' + dataset + '/dev.json'
    test_path='./data/'+dataset+'/test.json'
    ent2id_path='./data/'+dataset+'/entity2id.json'
    token2id_path = './data/'+dataset+'/token2id.json'
    save_path = './save_models/'+ dataset+'/best-model-without-global.bin'
    history_path='./saved_models/'+dataset+'/history.pickle'

    # bert_model = 'biobert-base-cased-v1.2'
    # bert_model='biobert-large-cased-v1.1'
    # bert_model = 'bert-base-cased'
    bert_model = 'bert-large-cased'
    bert_path = r'E:\pretraing_models\torch/' + bert_model
    bert_config_path = r'E:\pretraing_models\torch/' + bert_model + '/config.json'
    bert_vocab_path = r'E:\pretraing_models\torch/' + bert_model + '/vocab.txt'
    bert_checkpoint_path = r'E:\pretraing_models\torch/' + bert_model + '/pytorch_model.bin'
    word2vector = r'E:\pretraing_models\torch/glove.840b.cased/glove.840B.300d.txt'