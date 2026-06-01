"""
FULL autonomous pipeline (run in background, no interaction):
  1. Train 5 folds of Attention U-Net AND U-Net BCE (warm-start synthetic + Tversky).
     Same 5-fold split over the 10 real dev cases as the nnU-Net experiment.
  2. Build OOF dev probabilities (each dev case predicted by the fold that held it out).
  3. Build test probabilities (5-fold ensemble per net).
  4. Combine with nnU-Net (Tversky 5-fold) and select ensemble weights + threshold on
     OOF dev (NO leakage), then evaluate once on the 10 real test cases.
  5. Save everything.
"""
from __future__ import annotations
import glob, os, importlib.util, random, json, statistics as st, itertools
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEV = torch.device("cuda")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del repo
DATA = f"{ROOT}/data/Dataset_Preprocessed/Dataset_Hybrid"
OUT  = f"{ROOT}/models/custom_5fold"
NNU  = f"{ROOT}/data_probs"
TVT  = f"{NNU}/nnunet_test"        # nnU-Net test probs (key 'probabilities')
TVO  = f"{NNU}/nnunet_OOF_dev"     # nnU-Net OOF dev probs
EP, LR = 120, 1e-3

FOLDS = {0:["001","012"], 1:["004","018"], 2:["008","011"], 3:["002","017"], 4:["003","009"]}
DEV_IDS  = [c for v in FOLDS.values() for c in v]
TEST_IDS = ["005","006","007","010","013","014","015","016","019","020"]
NETS = {
  "attention": dict(file=f"{ROOT}/src/models/attention_unet3d.py", cls="AttentionUNet3D",
                    kw=dict(in_ch=1,base_ch=32,levels=4,use_checkpointing=True),
                    ckpt=f"{ROOT}/models/synthetic_base/Attention_UNet_Mixed_Size_best.pt"),
  "unet_bce": dict(file=f"{ROOT}/src/models/unet3d.py", cls="UNet3D",
                   kw=dict(in_ch=1,base_ch=32,levels=4),
                   ckpt=f"{ROOT}/models/synthetic_base/UNet_BCE_Mixed_Size_best.pt"),
}

def npy(cid,k): return np.load(f"{DATA}/real_dbt_{cid}/real_dbt_{cid}_{k}.npy").astype(np.float32)
def pad16(x,m=16):
    D,H,W=x.shape[-3:]; ru=lambda v:((v+m-1)//m)*m
    return torch.nn.functional.pad(x,(0,ru(W)-W,0,ru(H)-H,0,ru(D)-D))
def fresh_model(cfg):
    spec=importlib.util.spec_from_file_location("m",cfg["file"]); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    net=getattr(mod,cfg["cls"])(**cfg["kw"]).to(DEV)
    ck=torch.load(cfg["ckpt"],map_location=DEV,weights_only=False)
    net.load_state_dict(ck["model_state"] if "model_state" in ck else ck, strict=False)
    return net

class VolDS(Dataset):
    def __init__(self,ids,aug=True): self.ids=ids; self.aug=aug
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        cid=self.ids[i]; x=npy(cid,"img")[None]; y=(npy(cid,"mask")>0).astype(np.float32)[None]
        if self.aug and random.random()<0.5: x=x[:,:,:,::-1].copy(); y=y[:,:,:,::-1].copy()
        if self.aug and random.random()<0.5: x=x[:,:,::-1,:].copy(); y=y[:,:,::-1,:].copy()
        if self.aug: x=x*random.uniform(0.85,1.15)+random.uniform(-0.1,0.1)
        return torch.from_numpy(x), torch.from_numpy(y)

def tversky(p,y,a=0.3,b=0.7,e=1e-6):
    p=p.float(); y=y.float(); tp=(p*y).sum(); fp=(p*(1-y)).sum(); fn=((1-p)*y).sum()
    return 1.0-(tp+e)/(tp+a*fp+b*fn+e)
def bce_p(p,y,e=1e-6):
    p=torch.clamp(p.float(),e,1-e); return torch.nn.functional.binary_cross_entropy(p,y.float())

def train_fold(cfg, train_ids, tag):
    net=fresh_model(cfg); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP); ld=DataLoader(VolDS(train_ids),batch_size=1,shuffle=True)
    net.train()
    for ep in range(EP):
        for x,y in ld:
            x=pad16(x.to(DEV)); y=pad16(y.to(DEV)); opt.zero_grad()
            out=net(x); l=tversky(out,y)+bce_p(out,y); l.backward(); opt.step()
        sch.step()
        if ep%20==0 or ep==EP-1: print(f"    [{tag}] ep {ep+1}/{EP} loss={float(l):.4f}",flush=True)
    return net

@torch.no_grad()
def prob(net,cid):
    raw=npy(cid,"img"); D,H,W=raw.shape
    return net(pad16(torch.from_numpy(raw[None,None]).to(DEV)))[0,0,:D,:H,:W].cpu().numpy()

# ---- 1+2+3: train folds, collect OOF dev and test probs per net ----
dev_oof={}; test_prob={}
for name,cfg in NETS.items():
    print(f"=== NET {name}: training 5 folds ===",flush=True)
    test_acc={cid:[] for cid in TEST_IDS}; oof={}
    for f,valids in FOLDS.items():
        tr=[c for c in DEV_IDS if c not in valids]
        print(f"  fold {f}: train {len(tr)} / val {valids}",flush=True)
        net=train_fold(cfg,tr,f"{name}-f{f}")
        torch.save(net.state_dict(), f"{OUT}/pl_{name}_fold{f}.pth")
        for cid in valids: oof[cid]=prob(net,cid)        # OOF dev
        for cid in TEST_IDS: test_acc[cid].append(prob(net,cid))
        del net; torch.cuda.empty_cache()
    dev_oof[name]=oof
    test_prob[name]={cid:np.mean(test_acc[cid],0) for cid in TEST_IDS}
    print(f"  {name} done",flush=True)

# nnU-Net probs (Tversky 5-fold)
def alobj(a,shape):
    if a.shape==shape: return a
    t=np.transpose(a,(2,1,0)); return t if t.shape==shape else a
gt={cid:(npy(cid,"mask")>0) for cid in TEST_IDS}
gtdev={cid:(npy(cid,"mask")>0) for cid in DEV_IDS}
nn_test={cid: np.load(f"{TVT}/real_dbt_{cid}.npz")['probabilities'][1] for cid in TEST_IDS}
nn_dev ={cid: np.load(f"{TVO}/real_dbt_{cid}.npz")['probabilities'][1] for cid in DEV_IDS}

def dice(p,g): s=p.sum()+g.sum(); return 1.0 if s==0 else float(2*(p&g).sum()/(s+1e-8))
def largest(m):
    lab,n=label(m)
    if n==0: return m
    sz=np.bincount(lab.ravel()); sz[0]=0; return lab==sz.argmax()

def combo_eval(ids, probs_by_net, nn_probs, gtmap, w, thr, lc):
    wnn,wa,wb=w; res={}
    for cid in ids:
        shape=gtmap[cid].shape
        p=(wnn*alobj(nn_probs[cid],shape)+wa*alobj(probs_by_net["attention"][cid],shape)+
           wb*alobj(probs_by_net["unet_bce"][cid],shape))/(wnn+wa+wb)
        m=p>=thr; m=largest(m) if lc else m
        res[cid]=dice(m,gtmap[cid])
    return res

# ---- 4: select weights+threshold on OOF DEV, apply to TEST ----
print("\n=== selecting ensemble on OOF dev ===",flush=True)
grid_w=[(1,1,1),(2,1,1),(1,1,2),(2,1,2),(3,1,1),(1,2,1),(2,2,1)]
grid_t=[0.3,0.4,0.5]
best=None
for w in grid_w:
    for thr in grid_t:
        for lc in [False,True]:
            r=combo_eval(DEV_IDS, dev_oof, nn_dev, gtdev, w, thr, lc)
            m=st.mean(r.values())
            if best is None or m>best[0]: best=(m,w,thr,lc)
mdev,W,THR,LC=best
print(f"MEJOR en DEV: media={mdev:.4f}  w={W} thr={THR} largest={LC}",flush=True)

rt=combo_eval(TEST_IDS, test_prob, nn_test, gt, W, THR, LC)
xs=list(rt.values())
print("\n=== TEST (ensemble seleccionado en dev) ===",flush=True)
for cid in TEST_IDS: print(f"  real_dbt_{cid}: {rt[cid]:.3f}",flush=True)
print(f"=> media={st.mean(xs):.4f}  mediana={st.median(xs):.4f}  ceros={sum(1 for x in xs if x<1e-6)}/10",flush=True)
print(f"   (nnU-Net solo: 0.505)",flush=True)
json.dump({"weights":W,"thr":THR,"largest":LC,"dev_mean":mdev,
           "test_per_case":{f"real_dbt_{c}":rt[c] for c in TEST_IDS},
           "test_mean":st.mean(xs),"test_median":st.median(xs)},
          open(f"{OUT}/full_ensemble_result.json","w"),indent=2)
print("SAVED full_ensemble_result.json",flush=True)
