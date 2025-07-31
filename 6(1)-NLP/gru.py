import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate z_t = sigmoid(W_z x_t + U_z h_{t-1})
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)

        # Reset gate r_t = sigmoid(W_r x_t + U_r h_{t-1})
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidate hidden state n_t = tanh(W_h x_t + r_t * (U_h h_{t-1}))
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))    # update gate
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))    # reset gate
        n = torch.tanh(self.W_h(x) + r * self.U_h(h))   # candidate hidden state

        # 최종 hidden state 업데이트
        h_next = (1 - z) * n + z * h
        return h_next



class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

        # 마지막 hidden state를 입력 차원으로 
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, seq_len, _ = inputs.size()
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        for t in range(seq_len):
            h = self.cell(inputs[:, t, :], h)

        output = self.fc(h)
        return output