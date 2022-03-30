from transformers.models.bert.modeling_bert  import BertModel,BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertSelfAttention
from torch.nn.parameter import Parameter


class GlobalFeatureLearing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,  #B num_generated_triples H
        encoder_hidden_states,
        encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0] #hidden_states.shape
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :] # B 1 1 H
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0 #1 1 0 0 -> 0 0 -1000 -1000
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0] #B m H
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output) #B m H
        outputs = (layer_output,) + outputs
        return outputs


class EmbeddingLayer(torch.nn.Module):
    """Create embedding from glove.6b for LSTM module"""
    def __init__(self, cfg):
        super().__init__()
        from torch.nn.init import xavier_uniform_, kaiming_uniform_, xavier_normal_, kaiming_normal_, uniform_
        from tqdm import tqdm
        self.cfg = cfg
        self.embedding = torch.nn.Embedding(cfg.vocab_s, 300, 0)
        self.lookup_table = uniform_(torch.empty(cfg.vocab_s,
                                                    300),
                                                    a=-0.25,
                                                    b=0.25)
        self.dropout = nn.Dropout(0.2)
        token2id = cfg.token2id
        with open(self.cfg.word2vector, 'r', encoding='utf8') as f_in:
            num_pretrained_vocab = 0
            for line in tqdm(f_in):
                row = line.rstrip('\n').split(' ')
                if len(row) == 2:
                    assert int(row[1]) == 300, 'Pretrained dimension %d dismatch the setting %d' \
                                                         % (int(row[1]), 300)
                    continue
                if row[0] in token2id:
                    current_embedding = torch.FloatTensor([float(i) for i in row[1:]])
                    self.lookup_table[token2id[row[0]]] = current_embedding
                    num_pretrained_vocab += 1
        self.lookup_table[0] = 0.0
        self.embedding.weight.data.copy_(self.lookup_table)
        self.embedding.weight.requires_grad = True
        del self.lookup_table

    def __call__(self, vocab_id_list):
        embedding = self.embedding(vocab_id_list)
        return self.dropout(embedding)


class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class NerModel(BertPreTrainedModel):
    def __init__(self, config):
        super(NerModel, self).__init__(config)
        self.cfg = config
        if not config.use_bert:
            self.embed = EmbeddingLayer(config)
            self.layer_norm = nn.LayerNorm(config.hidden_size)
            self.spatial_drop = SpatialDropout(0.2)
            self.lstm = nn.LSTM(300, config.hidden_size//2, bidirectional=True, batch_first=True, dropout=0.2)
        else:
            self.bert = BertModel(config=config)
        if config.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.fc_b=nn.Linear(config.hidden_size,config.hidden_size)
        self.fc_e=nn.Linear(config.hidden_size,config.hidden_size)

        self.elu=nn.ELU()
        self.gelu = nn.GELU()
        self.fc_table = nn.Linear(config.hidden_size, config.num_enity)

        self.fc_b_g=nn.Linear(config.num_enity,config.hidden_size)
        self.fc_e_g=nn.Linear(config.num_enity,config.hidden_size)

        self.fc_gfl=GlobalFeatureLearing(config)

        self.u1 = Parameter(torch.Tensor(config.num_enity, config.hidden_size, config.hidden_size))
        self.u2 = Parameter(torch.Tensor(config.num_enity, 2*config.hidden_size))
        self.fc_affine = nn.Linear(config.hidden_size, config.num_enity * config.hidden_size, bias=False)
        self.fc_u2 = nn.Linear(2*config.hidden_size, config.num_enity, bias=False)

        torch.nn.init.orthogonal_(self.fc_b.weight, gain=1)
        torch.nn.init.orthogonal_(self.fc_e.weight, gain=1)
        torch.nn.init.orthogonal_(self.fc_table.weight, gain=1)
        torch.nn.init.orthogonal_(self.fc_b_g.weight, gain=1)
        torch.nn.init.orthogonal_(self.fc_e_g.weight, gain=1)

    def forward(self, token_ids, mask_token_ids):
        '''
        :param token_ids:
        :param token_type_ids:
        :param mask_token_ids:
        :param s_loc:
        :return: s_pred: [batch,seq,2]
        op_pred: [batch,seq,p,2]
        '''

        #encoder
        if self.cfg.use_bert:
            H=self.get_embed(token_ids, mask_token_ids)          #embed:BLH
        else:
            H = self.get_lstm_embed(token_ids)
        L=H.shape[1]

        # primary table generate
        H_B = self.fc_b(H) # BLH
        H_E= self.fc_e(H)

        h = self.elu(H_B.unsqueeze(2).repeat(1, 1, L, 1) * H_E.unsqueeze(1).repeat(1, L, 1, 1))  # BLLH
        B, L = h.shape[0], h.shape[1]
        primary_table = self.fc_table(h)  # BLLY

        if self.cfg.pred_mode:
            # global feature learing
            F_B = primary_table.max(dim=2).values  # BLY
            F_E = primary_table.max(dim=1).values  # BLY
            F_B_ = self.fc_b_g(F_B)
            F_E_ = self.fc_e_g(F_E)
            T_B = H_B + self.fc_gfl(F_B_, H, mask_token_ids)[0]
            T_E = H_E + self.fc_gfl(F_E_, H, mask_token_ids)[0]

            # prediction table generate
            h_ = self.elu(T_B.unsqueeze(2).repeat(1, 1, L, 1) * T_E.unsqueeze(1).repeat(1, L, 1, 1))
            pred_table = self.fc_table(h_)
            pred_table = pred_table.reshape([B, L, L, self.cfg.num_enity])
            return pred_table
        else:
            return primary_table

    def get_embed(self,token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed=bert_out[0]
        embed=self.dropout(embed)
        return embed

    def get_lstm_embed(self, token_ids):
        embeded = self.embed(token_ids)
        lstm_out = self.layer_norm(self.spatial_drop(self.lstm(embeded)[0]))
        return lstm_out


class NerRobertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert=RobertaModel.from_pretrained(config.bert_path)
        self.cfg = config
        if not config.use_bert:
            self.embed = EmbeddingLayer(config)
            self.layer_norm = nn.LayerNorm(config.hidden_size)
            self.spatial_drop = SpatialDropout(0.2)
            self.lstm = nn.LSTM(300, config.hidden_size//2, bidirectional=True, batch_first=True, dropout=0.2)
        if config.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.fc_b=nn.Linear(config.hidden_size,config.hidden_size)
        self.fc_e=nn.Linear(config.hidden_size,config.hidden_size)

        self.elu=nn.ELU()
        self.gelu = nn.GELU()
        self.fc_table = nn.Linear(config.hidden_size, config.num_enity)

        self.fc_b_g=nn.Linear(config.num_enity,config.hidden_size)
        self.fc_e_g=nn.Linear(config.num_enity,config.hidden_size)

        self.fc_gfl=GlobalFeatureLearing(config)

        self.u1 = Parameter(torch.Tensor(config.num_enity, config.hidden_size, config.hidden_size))
        self.u2 = Parameter(torch.Tensor(config.num_enity, 2*config.hidden_size))
        self.fc_affine = nn.Linear(config.hidden_size, config.num_enity * config.hidden_size, bias=False)
        self.fc_u2 = nn.Linear(2*config.hidden_size, config.num_enity, bias=False)

        torch.nn.init.orthogonal_(self.fc_b.weight, gain=1)
        torch.nn.init.orthogonal_(self.fc_e.weight, gain=1)
        torch.nn.init.orthogonal_(self.fc_table.weight, gain=1)
        torch.nn.init.orthogonal_(self.fc_b_g.weight, gain=1)
        torch.nn.init.orthogonal_(self.fc_e_g.weight, gain=1)

    def forward(self, token_ids, mask_token_ids):
        '''
        :param token_ids:
        :param token_type_ids:
        :param mask_token_ids:
        :param s_loc:
        :return: s_pred: [batch,seq,2]
        op_pred: [batch,seq,p,2]
        '''

        #encoder
        if self.cfg.use_bert:
            H=self.get_embed(token_ids, mask_token_ids)          #embed:BLH
        else:
            H = self.get_lstm_embed(token_ids)
        L=H.shape[1]

        # primary table generate
        H_B = self.fc_b(H) # BLH
        H_E= self.fc_e(H)

        h = self.elu(H_B.unsqueeze(2).repeat(1, 1, L, 1) * H_E.unsqueeze(1).repeat(1, L, 1, 1))  # BLLH
        B, L = h.shape[0], h.shape[1]
        primary_table = self.fc_table(h)  # BLLY

        #global feature learing
        F_B = primary_table.max(dim=2).values#BLY
        F_E = primary_table.max(dim=1).values#BLY
        F_B_ = self.fc_b_g(F_B)
        F_E_ = self.fc_e_g(F_E)
        T_B=H_B+self.fc_gfl(F_B_,H,mask_token_ids)[0]
        T_E=H_E+self.fc_gfl(F_E_,H,mask_token_ids)[0]

        # prediction table generate
        h_ = self.elu(T_B.unsqueeze(2).repeat(1, 1, L, 1) * T_E.unsqueeze(1).repeat(1, L, 1, 1))

        pred_table = self.fc_table(h_)

        return pred_table

    def get_embed(self,token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed=bert_out[0]
        embed=self.dropout(embed)
        return embed

    def get_lstm_embed(self, token_ids):
        embeded = self.embed(token_ids)
        lstm_out = self.layer_norm(self.spatial_drop(self.lstm(embeded)[0]))
        return lstm_out