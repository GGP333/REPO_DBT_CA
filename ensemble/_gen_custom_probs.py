"""Genera y guarda las probabilidades (test + OOF dev) de Attention U-Net y U-Net BCE
desde los modelos 5-fold guardados, para reproducir el ensemble sin reentrenar."""
import os, glob, importlib.util, numpy as np, torch
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.environ.get("DBT_DATA",f"{REPO}/data/Dataset_Preprocessed/Dataset_Both_RealWorld")
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDS={0:["001","012"],1:["004","018"],2:["008","011"],3:["002","017"],4:["003","009"]}
DEVIDS=[c for v in FOLDS.values() for c in v]
TEST=["005","006","007","010","013","014","015","016","019","020"]
def img(n): return np.load(f"{DATA}/{n}/{n}_img.npy").astype(np.float32)
def pad16(x,m=16):
    D,H,W=x.shape[-3:]; ru=lambda v:((v+m-1)//m)*m
    return torch.nn.functional.pad(x,(0,ru(W)-W,0,ru(H)-H,0,ru(D)-D))
def loadc(file,cls,kw,ck):
    s=importlib.util.spec_from_file_location("m",file); mod=importlib.util.module_from_spec(s); s.loader.exec_module(mod)
    net=getattr(mod,cls)(**kw).to(DEV); net.load_state_dict(torch.load(ck,map_location=DEV)); net.eval(); return net
@torch.no_grad()
def prob(net,n):
    raw=img(n); D,H,W=raw.shape
    return net(pad16(torch.from_numpy(raw[None,None]).to(DEV)))[0,0,:D,:H,:W].cpu().numpy()
NETS={"attention":(f"{REPO}/src/models/attention_unet3d.py","AttentionUNet3D",dict(in_ch=1,base_ch=32,levels=4,use_checkpointing=True),"pl_attention"),
      "unet_bce":(f"{REPO}/src/models/unet3d.py","UNet3D",dict(in_ch=1,base_ch=32,levels=4),"pl_unet_bce")}
for key,(file,cls,kw,pre) in NETS.items():
    os.makedirs(f"{REPO}/data_probs/{key}_test",exist_ok=True); os.makedirs(f"{REPO}/data_probs/{key}_OOF_dev",exist_ok=True)
    tacc={c:[] for c in TEST}; oof={}
    for f,val in FOLDS.items():
        net=loadc(file,cls,kw,f"{REPO}/models/custom_5fold/{pre}_fold{f}.pth")
        for c in val: oof[c]=prob(net,f"real_dbt_{c}")
        for c in TEST: tacc[c].append(prob(net,f"real_dbt_{c}"))
        del net; torch.cuda.empty_cache()
    for c in TEST: np.savez_compressed(f"{REPO}/data_probs/{key}_test/real_dbt_{c}.npz", prob=np.mean(tacc[c],0).astype(np.float16))
    for c in DEVIDS: np.savez_compressed(f"{REPO}/data_probs/{key}_OOF_dev/real_dbt_{c}.npz", prob=oof[c].astype(np.float16))
    print(f"{key}: probs guardadas ({len(TEST)} test + {len(oof)} oof)",flush=True)
print("DONE")
