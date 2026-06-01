"""
Sim->real fine-tuning for the custom 3D nets (Attention U-Net, U-Net BCE),
mirroring the validated nnU-Net pipeline.

  warm-start from the SYNTHETIC 'Mixed_Size' checkpoint  ->  fine-tune on 10 real dev
  cases (Tversky loss, low LR)  ->  evaluate per-case 3D Dice on 10 real test.

Models output PROBABILITIES (sigmoid inside), so losses operate on probs.
Run:  python ft_custom.py attention   |   python ft_custom.py unet_bce
"""
from __future__ import annotations
import sys, glob, os, random, importlib.util, json, statistics as st
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

NET = sys.argv[1] if len(sys.argv) > 1 else "attention"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEV = torch.device("cuda")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del repo
DATA = f"{ROOT}/data/Dataset_Preprocessed/Dataset_Hybrid"   # per-case real .npy
DEV_IDS  = ["001","002","003","004","008","009","011","012","017","018"]
TEST_IDS = ["005","006","007","010","013","014","015","016","019","020"]

CFG = {
    "attention": dict(file=f"{ROOT}/src/models/attention_unet3d.py",
                      cls="AttentionUNet3D", kw=dict(in_ch=1, base_ch=32, levels=4, use_checkpointing=True),
                      ckpt=f"{ROOT}/models/synthetic_base/Attention_UNet_Mixed_Size_best.pt"),
    "unet_bce": dict(file=f"{ROOT}/src/models/unet3d.py",
                     cls="UNet3D", kw=dict(in_ch=1, base_ch=32, levels=4),
                     ckpt=f"{ROOT}/models/synthetic_base/UNet_BCE_Mixed_Size_best.pt"),
}[NET]

def load_model():
    spec = importlib.util.spec_from_file_location("netmod", CFG["file"])
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    model = getattr(mod, CFG["cls"])(**CFG["kw"]).to(DEV)
    ck = torch.load(CFG["ckpt"], map_location=DEV, weights_only=False)
    sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"warm-start {CFG['ckpt'].split('/')[-3]} | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model

def vol(cid, kind):  # (D,H,W) float32 / mask
    return np.load(f"{DATA}/real_dbt_{cid}/real_dbt_{cid}_{kind}.npy").astype(np.float32)

class VolDS(Dataset):
    def __init__(self, ids, augment=False):
        self.ids = ids; self.augment = augment
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        cid = self.ids[i]
        x = vol(cid, "img")[None]                      # (1,D,H,W)
        y = (vol(cid, "mask") > 0).astype(np.float32)[None]
        if self.augment and random.random() < 0.5: x = x[:, :, :, ::-1].copy(); y = y[:, :, :, ::-1].copy()
        if self.augment and random.random() < 0.5: x = x[:, :, ::-1, :].copy(); y = y[:, :, ::-1, :].copy()
        if self.augment: x = x * random.uniform(0.85, 1.15) + random.uniform(-0.1, 0.1)
        return torch.from_numpy(x), torch.from_numpy(y)

def pad16(x, m=16):  # pad last 3 dims (D,H,W) up to multiples of m
    D, H, W = x.shape[-3:]
    def ru(v): return ((v + m - 1)//m)*m
    pad = (0, ru(W)-W, 0, ru(H)-H, 0, ru(D)-D)
    return torch.nn.functional.pad(x, pad)

def tversky_loss(p, y, a=0.3, b=0.7, eps=1e-6):  # on probabilities
    p = p.float(); y = y.float()
    tp = (p*y).sum(); fp = (p*(1-y)).sum(); fn = ((1-p)*y).sum()
    return 1.0 - (tp + eps)/(tp + a*fp + b*fn + eps)
def bce_p(p, y, eps=1e-6):
    p = torch.clamp(p.float(), eps, 1-eps)
    return torch.nn.functional.binary_cross_entropy(p, y.float())
def loss_fn(p, y): return tversky_loss(p, y) + bce_p(p, y)

def train(model, loader, epochs, lr, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    model.train()
    for ep in range(epochs):
        tot = 0.0; n = 0
        for x, y in loader:
            x = pad16(x.to(DEV)); y = pad16(y.to(DEV))
            opt.zero_grad(); out = model(x); l = loss_fn(out, y); l.backward(); opt.step()
            tot += float(l); n += 1
        sched.step()
        if ep % 5 == 0 or ep == epochs-1: print(f"  [{tag}] epoch {ep+1}/{epochs} loss={tot/n:.4f}", flush=True)

@torch.no_grad()
def evaluate(model, thr=0.5):
    model.eval(); res = {}
    for cid in TEST_IDS:
        raw = vol(cid, "img"); D, H, W = raw.shape
        x = pad16(torch.from_numpy(raw[None, None]).to(DEV))
        p = model(x)[0, 0, :D, :H, :W].cpu().numpy()
        g = vol(cid, "mask") > 0
        pb = p >= thr; s = pb.sum() + g.sum()
        res[f"real_dbt_{cid}"] = 1.0 if s == 0 else float(2*(pb & g).sum()/(s+1e-8))
    return res

if __name__ == "__main__":
    print(f"=== FT custom net: {NET} ===", flush=True)
    model = load_model()
    print("zero-shot (synthetic ckpt) on real test:", flush=True)
    r0 = evaluate(model); print(f"  mean={st.mean(r0.values()):.4f}", flush=True)
    train(model, DataLoader(VolDS(DEV_IDS, augment=True), batch_size=1, shuffle=True), epochs=120, lr=1e-3, tag="ft")
    res = evaluate(model); xs = list(res.values())
    for c in sorted(res): print(f"  {c}: {res[c]:.3f}")
    print(f"\n{NET} FT FINAL: mean={st.mean(xs):.4f} median={st.median(xs):.4f} zeros={sum(1 for x in xs if x<1e-6)}/10", flush=True)
    os.makedirs(f"{ROOT}/experiments/custom_nets_ft", exist_ok=True)
    json.dump({"net": NET, "per_case": res, "mean": st.mean(xs), "median": st.median(xs)},
              open(f"{ROOT}/experiments/custom_nets_ft/result_{NET}.json", "w"), indent=2)
    torch.save(model.state_dict(), f"{ROOT}/experiments/custom_nets_ft/model_{NET}.pth")
