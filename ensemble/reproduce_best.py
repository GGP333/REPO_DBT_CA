"""
Reproduce el MEJOR resultado: ensemble de 3 redes (nnU-Net + Attention U-Net + U-Net BCE).

No requiere GPU ni reentrenar: usa las probabilidades pre-computadas en data_probs/
y las máscaras reales en data/real/. Aplica los pesos seleccionados en validación
(OOF dev, sin leakage): nn=2, att=2, bce=3, umbral=0.3.

Resultado esperado: Dice medio ~0.518, mediana ~0.594, 0/10 casos nulos.

Uso:   python ensemble/reproduce_best.py
"""
import os, glob, numpy as np, statistics as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DP   = f"{REPO}/data_probs"
GT   = os.environ.get("DBT_DATA", f"{REPO}/data/real")
TEST = ["005","006","007","010","013","014","015","016","019","020"]

# Configuración ganadora (elegida en OOF dev, sin tocar test)
W = {"nn": 2, "att": 2, "bce": 3}
THR = 0.3

def nn_prob(c):  # nnU-Net guarda (2,Z,Y,X); canal 1 = tumor
    return np.load(f"{DP}/nnunet_test/real_dbt_{c}.npz")["probabilities"][1].astype(np.float32)
def cust_prob(sub, c):
    return np.load(f"{DP}/{sub}/real_dbt_{c}.npz")["prob"].astype(np.float32)
def mask(c):
    return (np.load(f"{GT}/real_dbt_{c}/real_dbt_{c}_mask.npy") > 0)
def dice(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2 * (p & g).sum() / (s + 1e-8))

print(f"Ensemble 3 redes  | pesos nn={W['nn']} att={W['att']} bce={W['bce']}  umbral={THR}\n")
print(f"{'caso':14s} {'Dice':>7s}")
scores = []
for c in TEST:
    p = (W["nn"]*nn_prob(c) + W["att"]*cust_prob("attention_test", c) + W["bce"]*cust_prob("unet_bce_test", c)) / sum(W.values())
    d = dice(p >= THR, mask(c))
    scores.append(d)
    print(f"real_dbt_{c}  {d:7.3f}")
print("-" * 24)
print(f"media   = {st.mean(scores):.4f}")
print(f"mediana = {st.median(scores):.4f}")
print(f"ceros   = {sum(1 for x in scores if x < 1e-6)}/10")
