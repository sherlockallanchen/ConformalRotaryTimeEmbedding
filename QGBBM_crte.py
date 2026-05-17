import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import gc
from tqdm import tqdm

SEED = 2026
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class PoetryCorpusDataset(Dataset):
    def __init__(self, lines: list, tokenizer: AutoTokenizer,
                 seq_len: int = 256, chunk_lines: int = 20000):
        self.seq_len = seq_len
        self.tokens = []
        for i in range(0, len(lines), chunk_lines):
            chunk_text = "\n".join(lines[i:i + chunk_lines]) + "\n"
            tok_chunk = tokenizer(chunk_text, return_tensors="pt",
                                  truncation=False)["input_ids"][0]
            self.tokens.append(tok_chunk)
            del chunk_text
            gc.collect()
        self.tokens = torch.cat(self.tokens)
        self.num_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + 1 + self.seq_len]
        return x, y

class ConformalRotaryTimeEmbedding(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 4096,
                 hubble_radius: float = 1024.0, base: float = 10000.0):
        super().__init__()
        self.hubble_radius = hubble_radius
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).float()
        eta = self.hubble_radius * torch.log(1.0 + t / self.hubble_radius)
        freqs = torch.outer(eta, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, seq_len: int):
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor):
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    """Gravitational Lensing with CRTE (Strictly Unitary)"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.nhead, self.dk).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.dk).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.dk).transpose(1, 2)

        # Apply CRTE: U(η) = exp(-iHη)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Standard attention without ad-hoc temperature scaling
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dk))
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class SpacetimeBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.ln1(x), mask, cos, sin)
        x = x + self.ff(self.ln2(x))
        return x


class QuantumGBBM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, d_ff: int = 512, dropout: float = 0.1,
                 max_len: int = 4096, hubble_radius: float = 1024.0,
                 rope_base: float = 10000.0,
                 bg_radiation_init: torch.Tensor | None = None):
        super().__init__()
        self.hubble_radius = hubble_radius

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.crte = ConformalRotaryTimeEmbedding(
            d_model // nhead,
            max_seq_len=max_len,
            hubble_radius=hubble_radius,
            base=rope_base,
        )
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SpacetimeBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, bias=False)

        # Weight tying: the Hilbert Space projector IS the embedding space
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.tok_emb.weight

        # Background Radiation (E_bg)
        if bg_radiation_init is not None:
            self.bg_radiation = nn.Parameter(bg_radiation_init.clone(),
                                             requires_grad=True)
        else:
            self.bg_radiation = nn.Parameter(torch.zeros(vocab_size),
                                             requires_grad=True)

        self.max_len = max_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T = x.size()
        assert T <= self.max_len, f"seq_len {T} > max_len {self.max_len}"

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        cos, sin = self.crte(T)

        h = self.drop(self.tok_emb(x))
        for block in self.blocks:
            h = block(h, causal_mask, cos, sin)
        h = self.ln_f(h)

        # E_θ = contextual projection + background radiation
        logits = self.out_proj(h) + self.bg_radiation
        return logits

    @torch.no_grad()
    def wave_function_collapse(self, idx, max_new_tokens,
                                temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def compute_metrics(model, loader, criterion):
    model.eval()
    total_ce = 0.0
    token_count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_ce += loss.item() * x.size(0) * x.size(1)
            token_count += x.size(0) * x.size(1)
    avg_ce = total_ce / token_count
    perplexity = math.exp(avg_ce)
    return avg_ce, perplexity


def evaluate_length_extrapolation(model, tokenizer, test_lines,
                                   train_seq_len: int = 256,
                                   eval_lengths: list | None = None,
                                   base_batch_size: int = 16):
    if eval_lengths is None:
        eval_lengths = [
            train_seq_len,
            train_seq_len * 2,
            train_seq_len * 3,
            train_seq_len * 4,
            train_seq_len * 6,
            train_seq_len * 8,
            train_seq_len * 12,
            train_seq_len * 16,
        ]

    criterion = nn.CrossEntropyLoss()
    results = {}
    R0 = model.hubble_radius
    eta_train = R0 * math.log(1.0 + train_seq_len / R0)

    print("\n" + "=" * 76)
    print("  Length Extrapolation Evaluation  (Q-GBBM with CRTE)")
    print("=" * 76)
    print(f"  Training seq_len  : {train_seq_len}")
    print(f"  Hubble Radius R₀  : {R0}")
    print(f"  Max eval length   : {max(eval_lengths)}")
    print("-" * 76)
    print(f"  {'Eval':>6} {'Raw':>6} {'Eff.η':>8} {'Eff.':>6} "
          f"{'CE':>10} {'PPL':>12} {'PPL Ratio':>10}")
    print(f"  {'Len':>6} {'Ratio':>6} {'max':>8} {'Ratio':>6} "
          f"{'':>10} {'':>12} {'':>10}")
    print("-" * 76)

    baseline_ppl = None

    for eval_len in eval_lengths:
        eval_batch = max(1,
            (base_batch_size * train_seq_len) // (eval_len * 2))

        test_ds = PoetryCorpusDataset(test_lines, tokenizer, seq_len=eval_len)
        test_loader = DataLoader(test_ds, batch_size=eval_batch,
                                 shuffle=False, drop_last=True)
        try:
            ce, ppl = compute_metrics(model, test_loader, criterion)
            raw_ratio = eval_len / train_seq_len
            eta_eval = R0 * math.log(1.0 + eval_len / R0)
            eff_ratio = eta_eval / eta_train

            if baseline_ppl is None:
                baseline_ppl = ppl
            ppl_ratio = ppl / baseline_ppl

            results[eval_len] = {
                "ce": ce, "ppl": ppl,
                "raw_ratio": raw_ratio,
                "eff_ratio": eff_ratio,
                "ppl_ratio": ppl_ratio
            }
            print(f"  {eval_len:>6} {raw_ratio:>5.0f}× "
                  f"{eta_eval:>7.1f}  {eff_ratio:>5.2f}× "
                  f"{ce:>10.4f} {ppl:>12.2f} {ppl_ratio:>9.4f}×")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  {eval_len:>6}   — OOM —")
                torch.cuda.empty_cache()
                results[eval_len] = None
            else:
                raise

        torch.cuda.empty_cache()
        gc.collect()

    print("-" * 76)

    print("\n  Summary of PPL Degradation:")
    for label, mult in [("2×", 2), ("4×", 4), ("8×", 8), ("16×", 16)]:
        key = train_seq_len * mult
        if key in results and results[key] is not None:
            print(f"    PPL@{label:>3} / PPL@1× = "
                  f"{results[key]['ppl_ratio']:.4f}")
        else:
            print(f"    PPL@{label:>3} / PPL@1× = — OOM —")

    baseline_results = {
        256:  {"ppl_ratio": 1.0000},
        512:  {"ppl_ratio": 1.3879},
        768:  {"ppl_ratio": 1.9466},
        1024: {"ppl_ratio": 2.3606},
        1536: {"ppl_ratio": 2.9036},
        2048: {"ppl_ratio": 3.2631},
        3072: {"ppl_ratio": 3.6560},
        4096: {"ppl_ratio": 3.8378},
    }

    print("\n  " + "=" * 72)
    print("  Comparison: Q-GBBM (CRTE) vs GPT-2 (Vanilla RoPE)")
    print("  " + "=" * 72)
    print(f"  {'Len':>6} {'Raw':>5} {'Eff.':>6} "
          f"{'GPT-2':>10} {'Q-GBBM':>10} {'Improvement':>12}")
    print(f"  {'':>6} {'Ratio':>5} {'η-Ratio':>6} "
          f"{'PPL×':>10} {'PPL×':>10} {'':>12}")
    print("  " + "-" * 72)

    for eval_len in eval_lengths:
        raw_r = eval_len / train_seq_len
        gpt2_r = baseline_results.get(eval_len, {}).get("ppl_ratio", None)
        qgbbm  = results.get(eval_len, None)

        if gpt2_r is not None and qgbbm is not None:
            q_r = qgbbm["ppl_ratio"]
            eff_r = qgbbm["eff_ratio"]
            improvement = (1 - q_r / gpt2_r) * 100
            print(f"  {eval_len:>6} {raw_r:>4.0f}×  {eff_r:>5.2f}× "
                  f"{gpt2_r:>9.4f}× {q_r:>9.4f}× "
                  f"{improvement:>+10.1f}%")

    print("  " + "-" * 72)
    print()
    return results

def train_and_evaluate_qgbbm(
    dataset_id: str = "biglam/gutenberg-poetry-corpus",
    epochs: int = 50,
    batch_size: int = 16,
    accum_steps: int = 4,
    lr: float = 3e-4,
    seq_len: int = 256,
    max_len: int = 4096,
    hubble_radius: float = 1024.0,
    rope_base: float = 10000.0,
):
    raw = load_dataset(dataset_id)
    split = raw["train"] if "train" in raw else next(iter(raw.values()))
    lines = list(split["line"])

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e18)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    random.shuffle(lines)

    n = len(lines)
    train_lines = lines[:int(0.8 * n)]
    val_lines   = lines[int(0.8 * n):int(0.9 * n)]
    test_lines  = lines[int(0.9 * n):]

    train_ds = PoetryCorpusDataset(train_lines, tokenizer, seq_len=seq_len)
    val_ds   = PoetryCorpusDataset(val_lines, tokenizer, seq_len=seq_len)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, generator=g)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, drop_last=True)

    vocab_size = len(tokenizer)
    token_counts = torch.bincount(train_ds.tokens,
                                   minlength=vocab_size).float()
    token_probs = (token_counts + 1) / (token_counts.sum() + vocab_size)
    bg_radiation_init = torch.log(token_probs)

    eta_train = hubble_radius * math.log(1 + seq_len / hubble_radius)
    eta_max   = hubble_radius * math.log(1 + max_len / hubble_radius)

    print(f"Vocab size       : {vocab_size}")
    print(f"Training tokens  : {len(train_ds.tokens):,}")
    print(f"Hubble Radius R₀ : {hubble_radius}")
    print(f"η(train_len={seq_len})  = {eta_train:.1f}  "
          f"(vanilla RoPE would be {seq_len})")
    print(f"η(eval_len={max_len})   = {eta_max:.1f}  "
          f"(vanilla RoPE would be {max_len})")
    print(f"Effective extrapolation at {max_len//seq_len}×: "
          f"{eta_max/eta_train:.2f}× "
          f"(vs {max_len/seq_len:.0f}× for vanilla RoPE)")

    model = QuantumGBBM(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        max_len=max_len,
        hubble_radius=hubble_radius,
        rope_base=rope_base,
        bg_radiation_init=bg_radiation_init,
    ).to(DEVICE)

    print(f"Parameters: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            betas=(0.9, 0.95), weight_decay=1e-4)

    total_steps = epochs * len(train_loader) // accum_steps
    warmup_steps = int(0.05 * total_steps)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            step / max(1, warmup_steps),
            0.5 * (1 + math.cos(math.pi * max(0, step - warmup_steps)
                                / max(1, total_steps - warmup_steps)))
        )
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0 \
               or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * accum_steps
            pbar.set_postfix(loss=f"{loss.item()*accum_steps:.4f}")

        torch.cuda.empty_cache()
        gc.collect()
        print(f"Epoch {epoch+1} | "
              f"Train CE: {epoch_loss/len(train_loader):.4f}")

    val_ce, val_ppl = compute_metrics(model, val_loader, criterion)
    print(f"\nValidation | CE: {val_ce:.4f} | PPL: {val_ppl:.2f}")

    extrapolation_results = evaluate_length_extrapolation(
        model, tokenizer, test_lines,
        train_seq_len=seq_len,
        base_batch_size=batch_size,
    )

    print("=" * 50)
    print("COLLAPSING THE WAVE FUNCTION")
    print("=" * 50 + "\n")

    seed_text = "Because I could not stop for death,"
    seed_ids = tokenizer.encode(seed_text, return_tensors="pt").to(DEVICE)

    for temp in [0.2, 0.5, 1.0]:
        print(f"--- Temperature: {temp} ---")
        generated = model.wave_function_collapse(
            seed_ids, max_new_tokens=500, temperature=temp, top_k=50
        )
        print(tokenizer.decode(generated[0].tolist()))
        print("\n")

    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "d_model": 256, "nhead": 8, "layers": 4, "d_ff": 512,
            "seq_len": seq_len, "max_len": max_len,
            "crte": True, "hubble_radius": hubble_radius,
            "rope_base": rope_base,
        },
        "val_ce": val_ce, "val_ppl": val_ppl,
        "extrapolation": {k: v for k, v in extrapolation_results.items()},
    }, "models/qgbbm_crte_lm_p100.pt")

    return extrapolation_results


if __name__ == "__main__":
    train_and_evaluate_qgbbm()




''' R_0=1024
----------------------------------------------------------------------------
    Eval    Raw    Eff.η   Eff.         CE          PPL  PPL Ratio
     Len  Ratio      max  Ratio
----------------------------------------------------------------------------
     256     1×   228.5   1.00×     4.3416        76.83    1.0000×
     512     2×   415.2   1.82×     4.6680       106.49    1.3860×
     768     3×   573.0   2.51×     5.0053       149.20    1.9419×
    1024     4×   709.8   3.11×     5.2095       183.00    2.3818×
    1536     6×   938.3   4.11×     5.4074       223.06    2.9033×
    2048     8×  1125.0   4.92×     5.5057       246.08    3.2030×
    3072    12×  1419.6   6.21×     5.6083       272.69    3.5493×
    4096    16×  1648.1   7.21×     5.6546       285.61    3.7174×
----------------------------------------------------------------------------

  Summary of PPL Degradation:
    PPL@ 2× / PPL@1× = 1.3860
    PPL@ 4× / PPL@1× = 2.3818
    PPL@ 8× / PPL@1× = 3.2030
    PPL@16× / PPL@1× = 3.7174
'''
