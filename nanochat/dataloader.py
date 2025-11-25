from collections import deque
import torch

from nanochat.common import get_dist_info
from nanochat.dataset import parquets_iter_batched 
from nanochat.tokenizer import get_tokenizer


def tokenizing_distributed_data_loader_with_state(
    B, T, split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None, 
):

    assert split in ["train", "val"]

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    def document_batches():
        pq_idx = 0
        rg_idx = 0

        for images_batch, texts_batch in parquets_iter_batched(
            subset="CoSyn_400k_chart",
            split=split,
            root="./finevision_data",
            start=ddp_rank,
            step=ddp_world_size,
            max_shards=max_shards,
        ):
            yield (images_batch, texts_batch), (pq_idx, rg_idx)
            rg_idx += 1

    batches = document_batches()

    needed_tokens = B * T + 1

    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    token_buffer = deque()

    while True:
        while len(token_buffer) < needed_tokens:
            (images_batch, texts_batch), (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(
                texts_batch,
                prepend=bos_token,
                num_threads=tokenizer_threads
            )
            for tokens in token_lists:
                token_buffer.extend(tokens)

        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        use_cuda = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long,
                               pin_memory=use_cuda)

        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]

        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda)

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}

        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader(*args, **kwargs):
    for inputs, targets, _ in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
