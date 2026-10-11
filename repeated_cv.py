"""Repeated stratified 5-fold CV (5 x 5 = 25 partitions) of the multimodal models.

Uses the unmodified model, training, FedAvg, client-partition and compression
functions of the original MultiModalMLP.py (copied as mm_original.py).
Federated models are evaluated after the final (100th) round; the best round is
recorded only for comparison with the single-split values in Fig. 6.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import mm_original as mm

R = r"D:\1-蜜蜂所读博文件\5-小可爱\5-论文数据\核磁、拉曼"
N_SPLITS, N_REPEATS, CV_SEED = 5, 5, 2026


def load(pairing):
    cache = os.path.join(HERE, f"data_{pairing}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["Xr"], d["Xn"], d["y"], list(d["names"])
    if pairing == "original":
        # exactly the matrices and shared label file used for modelling
        Xr = pd.read_excel(os.path.join(R, r"光谱数据\光谱数据.xlsx"), header=None).values.astype(np.float32)
        Xn = pd.read_excel(os.path.join(R, r"核磁数据\核磁数据.xlsx"), header=None).values.astype(np.float32)
        lab = pd.read_excel(os.path.join(R, r"光谱数据\分类.xlsx"), header=None).iloc[:, 0].astype(str).values
    else:
        # sensitivity check: pair rows within the same Chinese class label of each source sheet
        cr = pd.read_excel(os.path.join(R, "拉曼光谱、荔枝蜜130个样品-小可爱.xlsx"), sheet_name="Sheet1", header=None)
        cn = pd.read_excel(os.path.join(R, "核磁数据130个样品 - 副本(1).xlsx"), sheet_name="Sheet2", header=None)
        classes = ["成熟", "非成熟", "加热"]
        lr, ln = cr.iloc[:, 0].astype(str), cn.iloc[:, 0].astype(str)
        Xr_l, Xn_l, lab = [], [], []
        for c in classes:
            a = cr.loc[lr == c].iloc[:, 1:].astype(np.float32).values
            b = cn.loc[ln == c].iloc[:, 1:].astype(np.float32).values
            assert len(a) == len(b), (c, len(a), len(b))
            Xr_l.append(a); Xn_l.append(b); lab += [c] * len(a)
        Xr, Xn, lab = np.vstack(Xr_l), np.vstack(Xn_l), np.array(lab)
    names = sorted(set(lab))
    y = np.array([names.index(s) for s in lab], dtype=np.int64)
    np.savez(cache, Xr=Xr, Xn=Xn, y=y, names=np.array(names))
    return Xr, Xn, y, names


def predict(model, Xr, Xn, idx):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xr[idx]), torch.from_numpy(Xn[idx]))
    return logits.argmax(1).numpy()


def run_fed(ctor, client_loaders, client_weights, Xr, Xn, y, te, compress,
            num_rounds=100, local_epochs=3, lr=1e-3, sparsity=0.1, num_bits=8):
    """Mirror of mm.run_federated_experiment, additionally returning final-round predictions."""
    device = torch.device("cpu")
    global_model = ctor()
    global_state = global_model.state_dict()
    n_params = sum(p.numel() for p in global_model.parameters())
    scheme = "q_sparse" if compress else "none"
    hist, total_bits = [], 0.0
    for rnd in range(num_rounds):
        total_bits += n_params * 32.0 if scheme == "none" else n_params * sparsity * num_bits
        local_states = []
        for (dl, _, _), w in zip(client_loaders, client_weights):
            local_model = ctor()
            local_model.load_state_dict(global_state)
            sd_local = mm.train_local(local_model, dl, local_epochs, device, lr)
            delta = {k: sd_local[k] - global_state[k] for k in global_state.keys()}
            cdelta, bits_up = mm.compress_state_dict(delta, scheme=scheme, sparsity=sparsity, num_bits=num_bits)
            total_bits += bits_up
            local_states.append({k: global_state[k] + cdelta[k] for k in global_state.keys()})
        global_state = mm.fedavg(local_states, weights=client_weights)
        global_model.load_state_dict(global_state)
        hist.append(float(accuracy_score(y[te], predict(global_model, Xr, Xn, te))))
    return hist, predict(global_model, Xr, Xn, te), total_bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--pairing", default="original", choices=["original", "aligned"])
    ap.add_argument("--models", default="all", choices=["all", "mlp"])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    Xr, Xn, y, names = load(a.pairing)
    n_classes, dim_r, dim_n = len(names), Xr.shape[1], Xn.shape[1]
    ctors = [("MultiModalMLP", lambda: mm.MultiModalMLP(dim_r, dim_n, n_classes)),
             ("EarlyFusionMLP", lambda: mm.EarlyFusionMLP(dim_r, dim_n, n_classes)),
             ("MultiModalCNN1D", lambda: mm.MultiModalCNN1D(dim_r, dim_n, n_classes))]
    if a.models == "mlp":
        ctors = ctors[:1]
    parts = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                         random_state=CV_SEED).split(np.zeros(len(y)), y))

    for pid in range(a.worker, len(parts), a.nworkers):
        fout = os.path.join(a.out, f"partition_{pid:02d}.json")
        if os.path.exists(fout):
            continue
        tr, te = parts[pid]
        t0 = time.time()
        rec = dict(partition=pid, repeat=pid // N_SPLITS, fold=pid % N_SPLITS,
                   n_train=int(len(tr)), n_test=int(len(te)), test_idx=te.tolist(),
                   pairing=a.pairing, label_names=names, results=[])
        clients = mm.create_federated_clients(tr.copy(), n_clients=3, seed=0)
        sizes = [len(c["indices"]) for c in clients]
        weights = [s / sum(sizes) for s in sizes]

        def add(model, regime, pred, extra=None):
            r = dict(model=model, regime=regime,
                     accuracy=float(accuracy_score(y[te], pred)),
                     macro_f1=float(f1_score(y[te], pred, average="macro")),
                     pred=pred.tolist())
            if extra:
                r.update(extra)
            rec["results"].append(r)

        for m, (name, ctor) in enumerate(ctors):
            base = 100000 + 1000 * pid + 10 * m
            # centralized
            torch.manual_seed(base); np.random.seed(base)
            ds = mm.HoneySpectraDataset(Xr, Xn, y, indices=tr, has_raman=True, has_nmr=True)
            model = mm.train_centralized(ctor(), DataLoader(ds, batch_size=16, shuffle=True), 100,
                                         torch.device("cpu"), 1e-3)
            add(name, "Centralized", predict(model, Xr, Xn, te))
            # federated, uncompressed and compressed
            for k, compress in enumerate((False, True)):
                torch.manual_seed(base + 1 + k); np.random.seed(base + 1 + k)
                loaders = [(DataLoader(mm.HoneySpectraDataset(Xr, Xn, y, indices=c["indices"],
                                                             has_raman=c["has_raman"], has_nmr=c["has_nmr"]),
                                       batch_size=16, shuffle=True), c["has_raman"], c["has_nmr"])
                           for c in clients]
                hist, pred, bits = run_fed(ctor, loaders, weights, Xr, Xn, y, te, compress)
                add(name, "Fed-Comp" if compress else "Fed", pred,
                    dict(best_round_accuracy=max(hist), best_round=int(np.argmax(hist)) + 1,
                         last10_mean_accuracy=float(np.mean(hist[-10:])), history=hist,
                         comm_MB=bits / 8 / 1024 / 1024))
            print(f"[w{a.worker}] partition {pid} {name} done {time.time() - t0:.0f}s", flush=True)

        # random forest baseline on concatenated features (same settings as original)
        rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=0, n_jobs=a.threads)
        rf.fit(np.concatenate([Xr[tr], Xn[tr]], 1), y[tr])
        add("RandomForest", "Centralized", rf.predict(np.concatenate([Xr[te], Xn[te]], 1)))

        rec["seconds"] = time.time() - t0
        with open(fout, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        print(f"[w{a.worker}] partition {pid} saved ({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
