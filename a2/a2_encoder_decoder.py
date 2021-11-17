# Copyright 2020 University of Toronto, all rights reserved

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


class Encoder(EncoderBase):

    def init_submodules(self):
        # initialize parameterized submodules here: rnn, embedding
        # using: self.source_vocab_size, self.word_embedding_size, self.pad_id,
        # self.dropout, self.cell_type, self.hidden_state_size,
        # self.num_hidden_layers
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # relevant pytorch modules:
        # torch.nn.{LSTM, GRU, RNN, Embedding}
        self.embedding = torch.nn.Embedding(self.source_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)

        if self.cell_type.lower() == 'lstm':
            rnn = torch.nn.LSTM
        elif self.cell_type.lower() == 'gru':
            rnn = torch.nn.GRU
        else:
            rnn = torch.nn.RNN

        self.rnn = rnn(
            self.word_embedding_size, self.hidden_state_size, self.num_hidden_layers,
            bidirectional=True, dropout=self.dropout)

    def get_all_rnn_inputs(self, F):
        # compute input vectors for each source transcription.
        # F is shape (S, N)
        # x (output) is shape (S, N, I)
        x = self.embedding(F)
        return x

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # compute all final hidden states for provided input sequence.
        # make sure you handle padding properly!
        # x is of shape (S, N, I)
        # F_lens is of shape (N,)
        # h_pad is a float
        # h (output) is of shape (S, N, 2 * H)
        # relevant pytorch modules:
        # torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=h_pad)
        return output


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # initialize parameterized submodules: embedding, cell, ff
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # relevant pytorch modules:
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # torch.nn.{Embedding,Linear,LSTMCell,RNNCell,GRUCell}
        self.embedding = torch.nn.Embedding(self.target_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)

        if self.cell_type.lower() == 'lstm':
            rnn_cell = torch.nn.LSTMCell
        elif self.cell_type.lower() == 'gru':
            rnn_cell = torch.nn.GRUCell
        else:
            rnn_cell = torch.nn.RNNCell

        self.cell = rnn_cell(self.word_embedding_size, self.hidden_state_size)  # *2

        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)  # *2

    def get_first_hidden_state(self, h, F_lens):
        # build decoder's first hidden state. Ensure it is derived from encoder
        # hidden state that has processed the entire sequence in each
        # direction:
        # - Populate indices 0 to self.hidden_state_size // 2 - 1 (inclusive)
        #   with the hidden states of the encoder's forward direction at the
        #   highest index in time *before padding*
        # - Populate indices self.hidden_state_size // 2 to
        #   self.hidden_state_size - 1 (inclusive) with the hidden states of
        #   the encoder's backward direction at time t=0
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # htilde_tm1 (output) is of shape (N, 2 * H)
        # relevant pytorch modules: torch.cat
        h_temp = torch.empty(1, self.hidden_state_size)
        for i in range(F_lens.shape[0]):
            b = torch.cat((h[-1, i, :self.hidden_state_size//2], h[0, i, self.hidden_state_size//2:]), 0)\
                .view(1, self.hidden_state_size)  # F_lens[i]
            h_temp = torch.cat((h_temp, b), 0)

        htilde_tm1 = h_temp[1:]
        return htilde_tm1

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # determine the input to the rnn for *just* the current time step.
        # No attention.
        # E_tm1 is of shape (N,)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # xtilde_t (output) is of shape (N, Itilde)
        xtilde_t = self.embedding(E_tm1).to(h.device)
        xtilde_t[(E_tm1 == self.pad_id).nonzero().flatten()] = 0
        return xtilde_t

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # update the previous hidden state to the current hidden state.
        # xtilde_t is of shape (N, Itilde)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # htilde_t (output) is of same shape as htilde_tm1
        if self.cell_type == 'lstm':
            htilde_t, c = self.cell(xtilde_t, (htilde_tm1[0], htilde_tm1[1]))
            return (htilde_t, c)
        else:
            htilde_t = self.cell(xtilde_t, htilde_tm1)
            return htilde_t

    def get_current_logits(self, htilde_t):
        # determine un-normalized log-probability distribution over output
        # tokens for current time step.
        # htilde_t is of shape (N, 2 * H), even for LSTM (cell state discarded)
        # logits_t (output) is of shape (N, V)
        logits_t = self.ff(htilde_t)
        return logits_t


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # same as before, but with a slight modification for attention
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        self.embedding = torch.nn.Embedding(self.target_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)

        if self.cell_type.lower() == 'lstm':
            rnn_cell = torch.nn.LSTMCell
        elif self.cell_type.lower() == 'gru':
            rnn_cell = torch.nn.GRUCell
        else:
            rnn_cell = torch.nn.RNNCell

        self.cell = rnn_cell(self.word_embedding_size + self.hidden_state_size, self.hidden_state_size)  # * 2

        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)  # * 2

    def get_first_hidden_state(self, h, F_lens):
        # same as before, but initialize to zeros
        # relevant pytorch modules: torch.zeros_like
        # ensure result is on same device as h! # TODO
        # h = torch.empty(F_lens.size(0), self.hidden_state_size, device=h.device)
        return torch.zeros_like(h[0], device=h.device)

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # update to account for attention. Use attend() for c_t
        if self.cell_type == 'lstm':
            htilde_tm1 = htilde_tm1[0]
        x = self.embedding(E_tm1)
        x[(E_tm1 == self.pad_id).nonzero().flatten()] = 0
        c_t = self.attend(htilde_tm1, h, F_lens)
        xtilde_t = torch.cat((c_t, x), -1)
        return xtilde_t

    def attend(self, htilde_t, h, F_lens):
        # compute context vector c_t. Use get_attention_weights() to calculate
        # alpha_t.
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # c_t (output) is of shape (N, 2 * H)
        alpha_t = self.get_attention_weights(htilde_t, h, F_lens)
        c_t = alpha_t.unsqueeze(2) * h
        c_t = torch.sum(c_t, dim=0)
        return c_t

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of shape (S, N)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, N)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Determine energy scores via cosine similarity
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # e_t (output) is of shape (S, N)
        # htilde_t = htilde_t.permute(1, 0).unsqueeze(0)  # 1, 2*H, N
        # h = h.permute(0, 2, 1)
        htilde_t = htilde_t.unsqueeze(0)  # 1, 2*H, N
        return torch.nn.functional.cosine_similarity(htilde_t, h, dim=2, eps=1e-8)


####################################
#              BONUS               #
####################################
class DecoderWithLuongAttention(DecoderWithAttention):
    '''
    This class is for bonus. I implemented Luong attention decoder by reading his paper:
    "Effective Approaches to Attention-based Neural Machine Translation"
     a dot product between the decoder hidden state and a linear transform of the encoder state
    '''

    def init_submodules(self):
        # same as before, but with a slight modification for attention
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        self.embedding = torch.nn.Embedding(self.target_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)

        if self.cell_type.lower() == 'lstm':
            rnn_cell = torch.nn.LSTMCell
        elif self.cell_type.lower() == 'gru':
            rnn_cell = torch.nn.GRUCell
        else:
            rnn_cell = torch.nn.RNNCell

        self.cell = rnn_cell(self.word_embedding_size + self.hidden_state_size, self.hidden_state_size)

        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)
        self.attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_state_size)

    def forward(self, E_tm1, htilde_tm1, h, F_lens):
        self.check_input(E_tm1, htilde_tm1, h, F_lens)
        if htilde_tm1 is None:
            htilde_tm1 = self.get_first_hidden_state(h, F_lens)
            if self.cell_type == 'lstm':
                # initialize cell state with zeros
                htilde_tm1 = (htilde_tm1, torch.zeros_like(htilde_tm1))
        xtilde_t = self.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)
        h_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)
        if self.cell_type == 'lstm':
            h_att_t = self.get_attentional_hidden_state(h_t[0], h, F_lens)
        else:
            h_att_t = self.get_attentional_hidden_state(h_t, h, F_lens)
        logits_t = self.get_current_logits(h_att_t)
        return logits_t, h_t

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        xtilde_t = self.embedding(E_tm1)
        xtilde_t[(E_tm1 == self.pad_id).nonzero().flatten()] = 0
        return xtilde_t

    def get_attentional_hidden_state(self, htilde_t, h, F_lens):
        # htilde_t is of shape (N, 2 * H)
        # h_att_t (output) is of shape (N, 2*H) or (N, self.hidden_state_size)
        c_t = self.attend(htilde_t, h, F_lens)
        cat = torch.cat((c_t, htilde_t), 1)
        Wc_cat = self.attn(cat)
        h_att_t = torch.tanh(Wc_cat)
        return h_att_t

    def get_energy_scores(self, htilde_t, h):
        # Determine energy scores via dot product
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # e_t (output) is of shape (S, N)
        e_t = torch.bmm(h.permute(1, 0, 2), htilde_t.unsqueeze(2))
        return e_t.squeeze(2).permute(1, 0)


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # initialize the parameterized submodules: encoder, decoder
        # encoder_class and decoder_class inherit from EncoderBase and
        # DecoderBase, respectively.
        # using: self.source_vocab_size, self.source_pad_id,
        # self.word_embedding_size, self.encoder_num_hidden_layers,
        # self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        # self.target_vocab_size, self.target_eos
        # Recall that self.target_eos doubles as the decoder pad id since we
        # never need an embedding for it
        self.encoder = encoder_class(self.source_vocab_size,
                                     self.source_pad_id,
                                     self.word_embedding_size,
                                     self.encoder_num_hidden_layers,
                                     self.encoder_hidden_size,
                                     self.encoder_dropout,
                                     self.cell_type)

        self.decoder = decoder_class(self.target_vocab_size,
                                     self.target_eos,
                                     self.word_embedding_size,
                                     self.encoder_hidden_size*2,
                                     self.cell_type)

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # get logits over entire E. logits predict the *next* word in the
        # sequence.
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # E is of shape (T, N)
        # logits (output) is of shape (T - 1, N, Vo)
        # relevant pytorch modules: torch.{zero_like,stack}
        # hint: recall an LSTM's cell state is always initialized to zero.
        # Note logits sequence dimension is one shorter than E (why?)
        # print(htilde_tm1.shape)
        # if self.cell_type == 'lstm':
        #     htilde_tm1 = (htilde_tm1, torch.zeros_like(htilde_tm1))
        # htilde_tm1 = None
        # logits = []
        # for t in range(1, E.size(0)):  #TODO not sure about T-1
        #     logits_t, htilde_tm1 = self.decoder(E[t, :], htilde_tm1, h, F_lens)
        #     logits.append(logits_t)
        #
        # logits = torch.stack(logits)
        # print(logits.shape)
        # return logits
        htilde_tm1 = None

        logits_all = None
        T = E.shape[0]
        for timestep in range(T - 1):
            E_tm1 = E[timestep, :]
            logits_t, htilde_tm1 = self.decoder(E_tm1, htilde_tm1, h, F_lens)  # fix for lstm?
            if logits_all is None:
                logits_all = logits_t.unsqueeze(0)
            else:
                logits_all = torch.cat((logits_all, logits_t.unsqueeze(0)), dim=0)
        # print(logits_all.shape)
        return logits_all

    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        # htilde_t is of shape (N, K, 2 * H) or a tuple of two of those (LSTM)
        # logpb_tm1 is of shape (N, K)
        # b_tm1_1 is of shape (t, N, K)
        # b_t_0 (first output) is of shape (N, K, 2 * H) or a tuple of two of
        #                                                         those (LSTM)
        # b_t_1 (second output) is of shape (t + 1, N, K)
        # logpb_t (third output) is of shape (N, K)
        # relevant pytorch modules:
        # torch.{flatten,topk,unsqueeze,expand_as,gather,cat}
        # hint: if you flatten a two-dimensional array of shape z of (A, B),
        # then the element z[a, b] maps to z'[a*B + b]
        if self.cell_type == 'lstm':
            htilde_t = htilde_t[0]
        extensions_t = logpb_tm1.unsqueeze(-1) + logpy_t
        extensions_t = extensions_t.view((extensions_t.shape[0], -1))
        logpb_t, v = extensions_t.topk(self.beam_width, -1, largest=True, sorted=True)
        selected_paths = torch.div(v, logpy_t.size(-1))
        v = torch.remainder(v, logpy_t.size(-1))
        b_tm1_1 = b_tm1_1.gather(2, selected_paths.unsqueeze(0).expand_as(b_tm1_1))
        b_t_0 = htilde_t.gather(1, selected_paths.unsqueeze(-1).expand_as(htilde_t))
        b_t_1 = torch.cat([b_tm1_1, v.unsqueeze(0)], dim=0)
        if self.cell_type == 'lstm':
            b_t_0 = (b_t_0, b_t_0)
        return b_t_0, b_t_1, logpb_t
        # assert self.beam_width == 1, "Greedy requires beam width of 1"
        # extensions_t = (logpb_tm1.unsqueeze(-1) + logpy_t).squeeze(1)  # (N, V)
        # logpb_t, v = extensions_t.max(1)  # (N,), (N,)
        # logpb_t = logpb_t.unsqueeze(-1)  # (N, 1) == (N, K)
        # # v indexes the maximal element in dim=1 of extensions_t that was
        # # chosen, which equals the token index v in k -> v
        # v = v.unsqueeze(0).unsqueeze(-1)  # (1, N, 1) == (1, N, K)
        # b_t_1 = torch.cat([b_tm1_1, v], dim=0)
        # # For greedy search, all paths come from the same prefix, so
        # b_t_0 = htilde_t
        # return b_t_0, b_t_1, logpb_t
