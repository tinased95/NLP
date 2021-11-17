import a2_bleu_score
import a2_encoder_decoder
import torch
import a2_training_and_testing
#
# reference = '''\
# it is a guide to action that ensures that the military will always heed party commands'''.strip().split()
#
# candidate = '''\
# it is a guide to action which ensures that the military always obeys the commands of the party'''.strip().split()
#
#
# print(a2_bleu_score.n_gram_precision(reference, candidate, 1))
# print(a2_bleu_score.n_gram_precision(reference, candidate, 2))
# print(a2_bleu_score.brevity_penalty(reference, candidate))
# print(a2_bleu_score.BLEU_score(reference, candidate, 2))


def test_encoder_embedding():
    S = 3
    N = 7
    H = 512
    I = 8  # size of per_word input vector (embed dim)
    F = torch.ones(S, N, dtype=torch.long)
    F = torch.LongTensor(F)
    F_lens = torch.tensor([3, 2, 1, 1, 1, 2, 3])
    encoder = a2_encoder_decoder.Encoder(source_vocab_size=5, word_embedding_size=8)
    x = encoder.get_all_rnn_inputs(F)
    print("x: ", x.shape)
    h = encoder.get_all_hidden_states(x, F_lens, h_pad=1)
    print("h: ", h.shape)


def test_decoder_wo_attention_get_first():
    # N, V, H, S = 10, 94, 512, 12
    S = 4  # seq_len
    N = 7  # batch_size
    H = 2  # hidden_state_size of encoder
    h = torch.ones(S, N, H, dtype=torch.float)
    h = torch.FloatTensor(h)
    F_lens = torch.tensor([3, 2, 1, 1, 1, 2, 3])
    E_tm1 = torch.tensor([93, 93, 3, 3, 93, 3, 3])

    # decoder = a2_encoder_decoder.DecoderWithoutAttention(target_vocab_size=94, hidden_state_size=H,
    #                                                      word_embedding_size=2, cell_type='rnn')
    # htilde_tm1 = decoder.get_first_hidden_state(h, F_lens)  # h, F_lens
    # print(htilde_tm1.shape)
    # xtilde_t = decoder.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)
    # print("xtilde_t: ", xtilde_t.shape)
    # htilde_t = decoder.get_current_hidden_state(xtilde_t, htilde_tm1)
    # print(htilde_t.shape)
    # logits_t = decoder.get_current_logits(htilde_t)
    # print(logits_t.shape)

    decoder = a2_encoder_decoder.DecoderWithAttention(target_vocab_size=94, hidden_state_size=H,
                                                         word_embedding_size=2, cell_type='rnn')
    htilde_tm1 = decoder.get_first_hidden_state(h, F_lens)  # h, F_lens
    print(htilde_tm1.shape)
    xtilde_t = decoder.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)
    print("xtilde_t: ", xtilde_t.shape)
    htilde_t = decoder.get_current_hidden_state(xtilde_t, htilde_tm1)
    print(htilde_t.shape)
    logits_t = decoder.get_current_logits(htilde_t)
    print(logits_t.shape)
    att = decoder.get_attention_weights(htilde_t, h, F_lens)
    print(att)


def test_train_test():
    T = 7
    T_prim = 8
    N = 3
    target_sos = 0
    target_eos = 2
    E_ref = torch.LongTensor(torch.ones(T, N, dtype=torch.long))
    E_cand = torch.LongTensor(torch.ones(T_prim, N, dtype=torch.long))
    E_ref[0] = 0
    E_ref[6] = 2
    E_cand[0] = 0
    E_cand[1] = 3
    E_cand[6] = 2
    E_cand[7] = 2
    bleus = a2_training_and_testing.compute_batch_total_bleu(E_ref,
                                                             E_cand,
                                                             target_sos,
                                                             target_eos)
    print(bleus)

# test_encoder_embedding()
# test_decoder_wo_attention_get_first()
test_train_test()