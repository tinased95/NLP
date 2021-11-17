import torch
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)

import a2_encoder_decoder
from rnntest import random_batch
import a2_abcs

N = 100  # batch_size

input_var, input_lengths, target_var, target_lengths = random_batch(N)

input_lengths = torch.LongTensor(input_lengths).to(device)
input_var = input_var.to(device)

target_lengths = torch.LongTensor(target_lengths).to(device)
target_var = target_var.to(device)

print("F(SxN)", input_var.shape)
print("F_lens:(N,)", input_lengths.shape)

print("target(N)", target_var.shape)

encoder_test = a2_encoder_decoder.Encoder(source_vocab_size=512, word_embedding_size=8)
encoder_test.to(device)
print("hidden state size:", encoder_test.hidden_state_size)
hidden_concat = encoder_test(input_var, input_lengths, 0)
print("hidden encoder:", hidden_concat.shape)

# N = 100
# S = 12  # seq_len
#
# H = 1024  # hidden_state_size of encoder
# h = torch.empty(S, N, H, dtype=torch.float)
# h = torch.FloatTensor(h)
# # htilde_tm1 = torch.empty(N, int(H/2), dtype=torch.float)
# # htilde_tm1 = torch.FloatTensor(htilde_tm1)
#
# E_tm1 = torch.ones(N, dtype=torch.long)
# E_tm1 = torch.LongTensor(E_tm1)
#
# print(E_tm1.shape)
#
# decoder_test = a2_encoder_decoder.DecoderWithoutAttention(target_vocab_size=4, hidden_state_size=int(H/2), cell_type='rnn')
# decoder_test.to(device)
# htilde_tm1 = decoder_test.get_first_hidden_state(h, input_lengths)
# xtilde_t = decoder_test.get_current_rnn_input(E_tm1, htilde_tm1, h, input_lengths) # E_tm1, htilde_tm1, h, F_lens
# htilde_t = decoder_test.get_current_hidden_state(xtilde_t, htilde_tm1)
# logits_t = decoder_test.get_current_logits(htilde_t)
# print(xtilde_t.shape)
# print(htilde_t.shape)
# print(logits_t.shape)
#
# decoder_test = a2_encoder_decoder.DecoderWithAttention(target_vocab_size=4, hidden_state_size=int(H/2), cell_type='rnn')
# decoder_test.to(device)
# # h = decoder_test.get_first_hidden_state(h, input_lengths)
#
# htilde_t = torch.empty(N, H, dtype=torch.float)
# htilde_t = torch.FloatTensor(htilde_t)
#
# scores = decoder_test.get_energy_scores(htilde_t, h)
# print(scores.shape)