import os, json, argparse, random
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dgl.dataloading import GraphDataLoader
from graph import load_split, infer_dims
from model import EGATClassifier, GAT_LightningModule 
from config import default_output_dir, get_dataset_params
from utils import _set_seed


def compute_alpha_nodegrad(device, gat_model, graph):
    """
    α per node = L1 norm of grad of predicted-class prob w.r.t initial node features (node_f).
    Returns: alpha (r,), order_idx (desc), node_f_base (r, d)
    """
    node_f_base = graph.ndata['feat'].float().to(device)
    node_f = node_f_base.clone().detach().requires_grad_(True)
    preds, _, _ = gat_model(graph, node_f)                 # (1, C)
    y_hat = preds[0].argmax().item()
    target_prob = F.softmax(preds[0], dim=0)[y_hat] 
    grads = torch.autograd.grad(target_prob, node_f, create_graph=False)[0]  # (r, d)
    alpha = grads.abs().sum(dim=1).detach()                # (r,)
    order_vals, order_idx = torch.sort(alpha, descending=True)
    return alpha, order_idx, node_f_base


@torch.no_grad()
def _forward_with_masked_nodef(gat_model, graph, node_f_base, keep_idx):
    """Zero all nodes except keep_idx, then forward."""
    nf = torch.zeros_like(node_f_base)
    if keep_idx.numel() > 0:
        nf[keep_idx] = node_f_base[keep_idx]
    logits, _, _ = gat_model(graph, nf)
    return logits.squeeze(0)  # (C,)

@torch.no_grad()
def fidelity_input_perturb(device, gat_model, graph, node_f_base, order_idx, frk_step=3):
    """
    peletion/insertion fidelity by perturbing input node features (node_f).
    fractions are evaluated at {0, 1/frk_step, 2/frk_step, ..., 1}.
    Returns:
      fracs (len=frk_step+1), deletion_probs, insertion_probs, auc_del, auc_ins, p_full, p_empty
    """
    r = node_f_base.size(0)

    # predicted class from the full (unmasked) forward
    logits_full, _, _ = gat_model(graph, node_f_base)    # (1, C)
    y_hat = logits_full[0].argmax().item()

    # Baselines
    p_full  = F.softmax(logits_full[0], dim=0)[y_hat].item()
    logits_empty = _forward_with_masked_nodef(gat_model, graph, node_f_base, keep_idx=torch.tensor([], device=device, dtype=torch.long))
    p_empty = F.softmax(logits_empty, dim=0)[y_hat].item()

    # Fractions and the corresponding k's
    fracs = np.linspace(0.0, 1.0, frk_step + 1)  # target x-axis
    deletion_probs, insertion_probs = [], []

    for i in range(frk_step + 1):
        # k = round(i * r / frk_step) nodes to remove/add
        k = int(round(i * r / frk_step))

        # deletion: remove top-k
        keep_del = order_idx[k:] if k < r else torch.tensor([], device=device, dtype=torch.long)
        log_del = _forward_with_masked_nodef(gat_model, graph, node_f_base, keep_idx=keep_del)
        p_del = F.softmax(log_del, dim=0)[y_hat].item()
        deletion_probs.append(p_del)

        # insertion: start from none, add top-k
        keep_ins = order_idx[:k] if k > 0 else torch.tensor([], device=device, dtype=torch.long)
        log_ins = _forward_with_masked_nodef(gat_model, graph, node_f_base, keep_idx=keep_ins)
        p_ins = F.softmax(log_ins, dim=0)[y_hat].item()
        insertion_probs.append(p_ins)

    deletion_probs  = np.asarray(deletion_probs)
    insertion_probs = np.asarray(insertion_probs)

    # AUC on the uniform fraction grid
    auc_del = np.trapz(deletion_probs,  fracs)
    auc_ins = np.trapz(insertion_probs, fracs)

    return fracs, deletion_probs, insertion_probs, auc_del, auc_ins, p_full, p_empty

def compute_AUC_curve(device, gat_model, dataset, FRAC_STEP=3):
    '''
    compute AUC insertion/deletion curve
    fraction step: 1/FRAC_STEP (e.g., 3 -> {0, 1/3, 2/3, 1})
    dataset: graph dataset
    returns: lists (deletion_probs, insertion_probs)
    '''

    # run over all samples and aggregate on the same fraction
    all_auc_del, all_auc_ins = [], []
    grid = np.linspace(0.0, 1.0, FRAC_STEP + 1)
    avg_del, avg_ins, n_ok = np.zeros_like(grid), np.zeros_like(grid), 0

    per_image = []  # optional: store per-image metrics

    for index, graph_data in enumerate(dataset):
        graph = graph_data[0].to(device)
        try:
            alpha, order_idx, node_f_base = compute_alpha_nodegrad(device, gat_model, graph)

            fr, delp, insp, aucd, auci, p_full, p_empty = fidelity_input_perturb(
                device, gat_model, graph, node_f_base, order_idx, frk_step=FRAC_STEP
            )

            # accumulate, no interpolation needed: same grid for all.
            avg_del += delp
            avg_ins += insp
            all_auc_del.append(aucd); all_auc_ins.append(auci)
            n_ok += 1

            per_image.append({
                "AUC_del": aucd,
                "AUC_ins": auci,
                "p_full": p_full,
                "p_empty": p_empty
            })

        except Exception as e:
            print(f"[skip] {index}: {e}")

    if n_ok == 0:
        print("No data processed.")
        return [], []
    else:
        avg_del /= n_ok; avg_ins /= n_ok
        return all_auc_del, all_auc_ins


DATASETS = ["ph2", "derm7pt"]

def main():
    ap = argparse.ArgumentParser("conduct filedlity test")
    ap.add_argument("--output-dir", type=str, default=default_output_dir)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    args = ap.parse_args()


    for ds in DATASETS:
        print(f"\n==================== DATASET: {ds} ====================")
      
        # get dataset-specific parameters
        ds_params = get_dataset_params(ds) if ds else {}

        # only assign if dataset and key exists in config
        if ds_params:
            if "num_heads" in ds_params and args.num_heads==None:
                args.num_heads = ds_params["num_heads"]

            if "hidden_dim" in ds_params and args.hidden_dim==None:
                args.hidden_dim = ds_params["hidden_dim"]

        _set_seed(args.seed)
        device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

        # load data
        test_ds  = load_split(args.output_dir, ds, "test", device)
        in_dim, num_classes = infer_dims(test_ds)

        # GAT model
        model = EGATClassifier(
            in_feats=in_dim,
            out_feats=args.hidden_dim,
            num_heads=args.num_heads,
            out_dim=num_classes,
            feat_drop=0.0,
            node_drop=0.0,
        )

        # checkpoints / logs dir define
        current_dir = os.getcwd()
        args.output_dir = os.path.join(current_dir, args.output_dir)

        best_ckpt_path = os.path.join(args.output_dir, ds, "models", ds, "{}_best_model.ckpt".format(ds))
        gat_model = GAT_LightningModule.load_from_checkpoint(best_ckpt_path, model=model)

        # eval (AUC Curve)
        gat_model = gat_model.model.eval().to(device)
        all_auc_del, all_auc_ins = compute_AUC_curve(device, gat_model, test_ds)

        print(f"Mean AUC deletion:  {np.mean(all_auc_del):.4f} ± {np.std(all_auc_del):.4f}")
        print(f"Mean AUC insertion: {np.mean(all_auc_ins):.4f} ± {np.std(all_auc_ins):.4f}")
    
        
if __name__ == "__main__":
    main()