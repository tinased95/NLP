# Copyright 2020 University of Toronto, all rights reserved

'''Functions related to training and testing.

You don't need anything more than what's been imported here.
'''

import torch
import a2_bleu_score


from tqdm import tqdm


def train_for_epoch(model, dataloader, optimizer, device):
    '''Train an EncoderDecoder for an epoch

    An epoch is one full loop through the training data. This function:

    1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
       keeping track of what id the loss considers "padding"
    2. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E``)
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens`` and ``E``.
       2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
       3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
          probabilities.
       4. Modifies ``E`` for the loss function, getting rid of a token and
          replacing excess end-of-sequence tokens with padding using
        ``model.get_target_padding_mask()`` and ``torch.masked_fill``
       5. Flattens out the sequence dimension into the batch dimension of both
          ``logits`` and ``E``
       6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
       7. Calls ``loss.backward()`` to backpropagate gradients through
          ``model``
       8. Calls ``optim.step()`` to update model parameters
    3. Returns the average loss over sequences

    Parameters
    ----------
    model : EncoderDecoder
        The model we're training.
    dataloader : HansardDataLoader
        Serves up batches of data.
    device : torch.device
        A torch device, like 'cpu' or 'cuda'. Where to perform computations.
    optimizer : torch.optim.Optimizer
        Implements some algorithm for updating parameters using gradient
        calculations.

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''
    # If you want, instead of looping through your dataloader as
    # for ... in dataloader: ...
    # you can wrap dataloader with "tqdm":
    # for ... in tqdm(dataloader): ...
    # This will update a progress bar on every iteration that it prints
    # to stdout. It's a good gauge for how long the rest of the epoch
    # will take. This is entirely optional - we won't grade you differently
    # either way.
    # If you are running into CUDA memory errors part way through training,
    # try "del F, F_lens, E, logits, loss" at the end of each iteration of
    # the loop.
    padding = model.target_sos  # model.target_eos (99)or is it model.source_pad_id = target_sos (98)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=padding, reduction='sum')
    losses = 0
    num_seq = 0
    for (F, F_lens, E) in tqdm(dataloader):
        F = F.to(device)  # (S,N)
        F_lens = F_lens.to(device)  # (N)
        E = E.to(device)  # (T-1,N)
        optimizer.zero_grad()
        logits = model(F, F_lens, E)  # logits -> (T-1, N, V)

        # step 4
        E = E[1:, :]
        mask = model.get_target_padding_mask(E)  # ignoring SOS tokens for all seqs for the masking
        E = E.masked_fill(mask, padding)  # masking extra eos tokens, (T-1, N), T-1 because we removed SOS
        # step 5
        E = E.view(-1)
        logits = logits.view(-1, logits.shape[2])
        loss = criterion(logits, E)
        loss.backward()
        optimizer.step()
        losses = losses + loss.item()
        num_seq = num_seq + F_lens.shape[0]

        del F, F_lens, E, logits, loss

    avg_loss = (losses / num_seq if num_seq != 0 else 0)
    return avg_loss


def compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos):
    '''Compute the total BLEU score over elements in a batch

    Parameters
    ----------
    E_ref : torch.LongTensor
        A batch of reference transcripts of shape ``(T, N)``, including
        start-of-sequence tags and right-padded with end-of-sequence tags.
    E_cand : torch.LongTensor
        A batch of candidate transcripts of shape ``(T', N)``, also including
        start-of-sequence and end-of-sequence tags.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    total_bleu : float
        The sum total BLEU score for across all elements in the batch. Use
        n-gram precision 4.
    '''
    # you can use E_ref.tolist() to convert the LongTensor to a python list
    # of numbers
    n = 4  # ngrams
    total_bleu = 0
    for seq_num in range(min(E_ref.shape[1], E_cand.shape[1])):
        reference = E_ref[:, seq_num].squeeze(0)
        reference = reference[((reference != target_sos) & (reference != target_eos)).nonzero().squeeze()].tolist()
        # print(E_cand)
        candidate = E_cand[:, seq_num].squeeze(0)
        candidate = candidate[((candidate != target_sos) & (candidate != target_eos)).nonzero().squeeze()].tolist()
        if not isinstance(reference, list):
            reference = [reference]
        if not isinstance(candidate, list):
            candidate = [candidate]
        total_bleu += a2_bleu_score.BLEU_score(reference, candidate, n)
    return total_bleu


def compute_average_bleu_over_dataset(
        model, dataloader, target_sos, target_eos, device):
    '''Determine the average BLEU score across sequences

    This function computes the average BLEU score across all sequences in
    a single loop through the `dataloader`.

    1. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E_ref``):
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens``. No need for ``E_cand``, since it will always be
          compared on the CPU.
       2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
       3. Extracts the top path per beam as ``E_cand = b_1[..., 0]``
       4. Computes the total BLEU score of the batch using
          :func:`compute_batch_total_bleu`
    2. Returns the average per-sequence BLEU score

    Parameters
    ----------
    model : EncoderDecoder
        The model we're testing.
    dataloader : HansardDataLoader
        Serves up batches of data.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    avg_bleu : float
        The total BLEU score summed over all sequences divided by the number of
        sequences
    '''
    n_batch = 0
    total = 0
    for F, F_lens, E in tqdm(dataloader):
        F = F.to(device)
        F_lens = F_lens.to(device)
        b_1 = model(F, F_lens)
        E_cand = b_1[:, 0]
        total += compute_batch_total_bleu(E, E_cand, target_sos, target_eos)
        n_batch += 1
    return total / n_batch

