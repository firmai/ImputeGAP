import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from build.lib.imputegap.recovery.evaluation import Evaluation
from build.lib.imputegap.tools import utils
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 1000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

MASK_CHAR = "0.0"

# Load and preprocess data
torch.manual_seed(1337)
ts = TimeSeries()
ts.load_series(utils.search_path("chlorine"))
ts.normalize()
ctn = ts.Contamination.mcar(ts.data)
alhp = Imputation.Statistics.ZeroImpute(incomp_data=ctn)
alhp.impute()
text = alhp.recov_data

ts.plot(ts.data, ctn, text, display=True)

text = alhp.recov_data.astype(str)  # ensure all values are strings
text = text.flatten()          # convert to 1D array
chars = sorted(set(text))      # safe to use set now

vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode_with_mask(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

def get_masked_input_target(text, mask_prob=0.15):
    encoded = encode_with_mask(text)
    input_ids = torch.tensor(encoded, dtype=torch.long)
    target_ids = input_ids.clone()
    mask = torch.tensor([c == stoi[MASK_CHAR] for c in encoded])
    return input_ids.unsqueeze(0), target_ids.unsqueeze(0), mask.unsqueeze(0)

# Prepare data
masked_text = text  # already masked using np.nan to '_'
input_ids, target_ids, mask = get_masked_input_target(masked_text)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            if mask is not None:
                mask = mask.view(B * T)
                loss = F.cross_entropy(logits[mask], targets[mask])
            else:
                loss = F.cross_entropy(logits, targets)
        return logits, loss

def impute_missing(model, input_ids, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_ids.to(device), mask=mask.to(device))
        pred_ids = torch.argmax(logits, dim=-1)
        imputed = input_ids.clone()
        imputed[mask] = pred_ids[mask]
    return imputed

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    model.train()
    logits, loss = model(input_ids.to(device), targets=target_ids.to(device), mask=mask.to(device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        print(f"Step {iter}, Loss: {loss.item():.4f}")

# Impute and show results
imputed = impute_missing(model, input_ids, mask)

decoded_tokens = [itos[i] for i in imputed[0].tolist()]
original_tokens = [itos[i] for i in input_ids[0].tolist()]

print("Original:")
print(original_tokens)
print("\nImputed:")
print(decoded_tokens)

# Convert to float array
numeric_array = np.array([float(tok) if tok != MASK_CHAR else np.nan for tok in decoded_tokens])
print(f"{numeric_array.shape}")

# Evaluate
rmse = Evaluation(input_data=ts.data, incomp_data=text, recov_data=numeric_array)
