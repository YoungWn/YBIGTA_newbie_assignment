import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam
from transformers import PreTrainedTokenizer
from typing import Literal

# 구현하세요!
def build_training_data(
    token_ids: LongTensor,
    window: int
) -> tuple[LongTensor, LongTensor]:
    context_list = []
    center_tokens = token_ids[window: -window]
    
    for idx in range(window, len(token_ids) - window):
        context = token_ids[idx - window: idx].tolist() + token_ids[idx + 1: idx + window + 1].tolist()
        context_list.append(context)
    
    return torch.LongTensor(context_list), center_tokens


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        for epoch in range(num_epochs):
            for line in corpus:
                tokens = tokenizer(line, return_tensors="pt", truncation=True, padding=False)
                input_ids = tokens["input_ids"].squeeze(0)

                contexts, centers = build_training_data(input_ids, self.window_size)
                if contexts.size(0) == 0 or centers.size(0) == 0:
                    continue

                if self.method == "cbow":
                    self._train_cbow(contexts, centers, loss_fn, optimizer)
                else:
                    self._train_skipgram(contexts, centers, loss_fn, optimizer)

            print(f"Epoch [{epoch+1}/{num_epochs}] complete.")

    def _train_cbow(
        self,
        # 구현하세요!
        context_batch: LongTensor,
        target_batch: LongTensor,
        loss_fn: nn.Module,
        optimizer: Adam
    ) -> None:
        # 구현하세요!
        embedded = self.embeddings(context_batch)
        context_vector = embedded.mean(dim=1)
        pred = self.weight(context_vector)

        loss = loss_fn(pred, target_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _train_skipgram(
        self,
        # 구현하세요!
        context_batch: LongTensor,
        target_batch: LongTensor,
        loss_fn: nn.Module,
        optimizer: Adam
    ) -> None:
        # 구현하세요!
        center_emb = self.embeddings(target_batch)
        pred = self.weight(center_emb)

        total_loss = 0.0
        for i in range(context_batch.size(1)):
            context_words = context_batch[:, i]
            loss = loss_fn(pred, context_words)
            total_loss += loss

        avg_loss = total_loss / context_batch.size(1)
        avg_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
