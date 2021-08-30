import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel


class NERModelBase(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

        # Load BERT after initializing classifier weigths
        self.bert = BertModel.from_pretrained(config.model_name_or_path,
                                              output_attentions=config.output_attentions,
                                              output_hidden_states=config.output_hidden_states)
        if config.pretrained_frozen:
            print("Freezing BERT parameters")
            for param in self.bert.parameters():
                param.requires_grad = False

        if config.vocab_size != self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(config.vocab_size)

            if config.pretrained_frozen:
                print("[WARNING] New tokens have been added, but BERT won't be trainable")

    def forward_bert(self, input_ids, attention_mask, token_type_ids, position_ids=None, head_mask=None, inputs_embeds=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        if self.config.output_attentions:
            attentions = outputs[-1]
            return sequence_output, attentions
        else:
            return sequence_output

    def ner_loss(self, logits, labels, label_mask=None, token_weights=None):
        if labels is None:
            loss = torch.tensor(0, dtype=torch.float, device=logits.device)
        else:
            # Only keep active parts of the loss
            if label_mask is not None:

                if token_weights is not None:
                    loss_fct = nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                    loss = loss * label_mask.view(-1) * token_weights.view(-1)
                    loss = loss.sum() / label_mask.sum()
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    active_loss = label_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def forward(self, input_ids, attention_mask, token_type_ids,
                position_ids=None, labels=None, label_mask=None, head_mask=None, inputs_embeds=None,
                token_weights=None, wrap_scalars=False):
        raise NotImplementedError('The NERModelBase class should never execute forward')


class NERModel(NERModelBase):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask, token_type_ids,
                position_ids=None, labels=None, label_mask=None, head_mask=None, inputs_embeds=None,
                token_weights=None, wrap_scalars=False):

        sequence_output = self.forward_bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)

        logits = self.classifier(sequence_output)
        loss = self.ner_loss(logits, labels, label_mask, token_weights)

        # Dummy losses
        first_loss = torch.tensor(0., device=logits.device).float()
        second_loss = torch.tensor(0., device=logits.device).float()

        if wrap_scalars: # for parallel models
            loss = loss.unsqueeze(0)
            first_loss = first_loss.unsqueeze(0)
            second_loss = second_loss.unsqueeze(0)

        return [loss, first_loss, second_loss], logits


class NERDevlinModel(NERModelBase):
    def __init__(self, config):
        super().__init__(config)

        lstm_hidden_size = self.hidden_size
        if config.lstm_bidirectional:
            lstm_hidden_size = lstm_hidden_size // 2

        if self.config.use_lstm:
            self.blstm = nn.LSTM(input_size=self.hidden_size,
                                 hidden_size=lstm_hidden_size,
                                 num_layers=config.lstm_layers,
                                 bidirectional=config.lstm_bidirectional,
                                 dropout=config.lstm_dropout,
                                 batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids,
                position_ids=None, labels=None, label_mask=None, head_mask=None, inputs_embeds=None,
                token_weights=None, wrap_scalars=False):

        sequence_output = self.forward_bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)

        if self.config.use_lstm:
            sequence_output, _ = self.blstm(sequence_output)

        logits = self.classifier(sequence_output)

        loss = self.ner_loss(logits, labels, label_mask, token_weights)

        # Dummy losses
        first_loss = torch.tensor(0, device=logits.device)
        second_loss = torch.tensor(0, device=logits.device)

        if wrap_scalars: # for parallel models
            loss = loss.unsqueeze(0)
            first_loss = first_loss.unsqueeze(0)
            second_loss = second_loss.unsqueeze(0)

        return [loss, first_loss, second_loss], logits

