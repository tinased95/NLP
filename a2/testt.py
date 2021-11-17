import a2_encoder_decoder
import torch
import a2_abcs

#
# s = 3
# n = 7
# h = 5
#
# x = torch.randn(s, n, 2 * h)
# x[2, 3, :] = 0
# x[2, 2, :] = 0
# x[2, 1, :] = 0
# x[1, 3, :] = 0
# x[1, 2, :] = 0
#
# F_lens = torch.tensor([3, 2, 1, 1, 1, 2, 3])
# a = torch.empty(1, 2*h)
# for i in range(F_lens.shape[0]):
#     b = torch.cat((x[F_lens[i]-1, i, :h], x[0, i, h:]), 0).view(1, 2*h)
#     a = torch.cat((a, b), 0)
#
#
# print(a[1:].shape)



S = 10  # seq_len
N = 3  # batch_size
H = 512
h_encoder = torch.ones(S, N, 2*H, dtype=torch.float)
h_encoder = torch.FloatTensor(h_encoder)
F_lens = torch.ones(N, dtype=torch.long)
F_lens = torch.LongTensor(F_lens)
F_lens[0] = 10

# decoder = a2_encoder_decoder.DecoderWithoutAttention(target_vocab_size=10, hidden_state_size=H, cell_type='gru', word_embedding_size=8)
# htilde_0 = decoder.get_first_hidden_state(h_encoder, F_lens)
# x_tilde_t = decoder.get_current_rnn_input(F_lens, htilde_0, h_encoder, F_lens)
# print(x_tilde_t.shape)
# print(htilde_0.shape)
# print(decoder.cell_type)
# htilde_t = decoder.get_current_hidden_state(x_tilde_t, htilde_0)
# print(htilde_t.shape)
# logits_t = decoder.get_current_logits(htilde_t)
# print(logits_t.shape)

# decoder_test = a2_encoder_decoder.DecoderWithAttention(target_vocab_size=4, hidden_state_size=H,
#                                                        cell_type='rnn', word_embedding_size=6)
# decoder_test.init_submodules()
# htilde_t = torch.empty(N, 2* H, dtype=torch.float)
# htilde_t = torch.FloatTensor(htilde_t)
# h = decoder_test.get_first_hidden_state(h_encoder, F_lens)
# # print(h.shape)
# # htilde_t = torch.empty(N, 2*H, dtype=torch.float)
# # htilde_t = torch.FloatTensor(htilde_t)
# # print(htilde_t.shape)
# # print(h.shape)
# e_t = decoder_test.get_energy_scores(htilde_t, h)
# c_t = decoder_test.attend(htilde_t, h, F_lens)
# # print(c_t.shape)
# xtilde_t = decoder_test.get_current_rnn_input(F_lens, htilde_t, h, F_lens)
# # print(x.shape)
# htilde_t = decoder_test.get_current_hidden_state(xtilde_t, htilde_t)
# print(htilde_t.shape)

# encoder_test = a2_encoder_decoder.Encoder(source_vocab_size=512, word_embedding_size=8)
encoderdecoder_test = a2_encoder_decoder.EncoderDecoder(a2_encoder_decoder.Encoder,
                                                        a2_encoder_decoder.DecoderWithoutAttention,
                                                        source_vocab_size=4,
                                                        target_vocab_size=5,
                                                        cell_type='lstm',
                                                        beam_width=4)
T = 9
# E = torch.ones(T, N, dtype=torch.long)
# E = torch.LongTensor(E).random_(10)
# print(E)
E = torch.zeros(T, N, dtype=torch.long)
E = torch.LongTensor(E).random_(0, 5)
# print(E)
logits = encoderdecoder_test.get_logits_for_teacher_forcing(h_encoder, F_lens, E)
# print(logits.shape)

htilde_t = torch.FloatTensor(torch.zeros(N, encoderdecoder_test.beam_width, 2 * encoderdecoder_test.encoder_hidden_size, dtype=torch.float))
b_tm1_1 = torch.LongTensor(torch.zeros(T, N, encoderdecoder_test.beam_width, dtype=torch.long))
logpb_tm1 = torch.FloatTensor(torch.zeros(N, encoderdecoder_test.beam_width, dtype=torch.float))
logpy_t = torch.FloatTensor(torch.zeros(N, encoderdecoder_test.beam_width, encoderdecoder_test.target_vocab_size, dtype=torch.float))
b_t_0, b_t_1, logpb_t = encoderdecoder_test.update_beam(htilde_t, b_tm1_1, logpb_tm1, logpy_t)
# print(encoderdecoder_test.target_vocab_size)
# print(b_t_0.shape)
# print(b_t_1.shape)
# print(logpb_t.shape)