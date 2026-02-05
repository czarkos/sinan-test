#!/usr/bin/env python3
"""
Generate surrogate training data by running the BNN (teacher) on the dataset.
Saves (X, Y_bnn) where X is scaled full-dim input and Y_bnn is BNN mean
prediction (first 2 outputs) in original scale.

Fits scalers from train data so feature dims match the BNN checkpoint (1005).
"""

import argparse
import os
import sys
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler

# Add ml_docker_swarm to path for BayesianMLP import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_docker_swarm'))
from train_bnn_explore import BayesianMLP


def load_and_reshape(filepath):
    arr = np.load(filepath)
    return arr.reshape(arr.shape[0], -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with sys_data_train.npy, lat_data_*.npy, nxt_k_*.npy, nxt_k_*_label.npy")
    parser.add_argument("--bnn-model", type=str, required=True,
                        help="Path to BNN .pth checkpoint")
    parser.add_argument("--out-dir", type=str, default="data",
                        help="Directory to save X_surrogate_*.npy, Y_bnn_*.npy (and optional scalers)")
    parser.add_argument("--mc-samples", type=int, default=50,
                        help="Number of MC samples for BNN mean prediction")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch size for BNN inference")
    parser.add_argument("--save-scalers", action="store_true",
                        help="Save fitted scalers to out-dir for use by predictor with this surrogate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir

    # Infer BNN arch from checkpoint
    sd = torch.load(args.bnn_model, map_location="cpu")
    input_dim = sd["weight_mus.0"].shape[1]
    output_dim = sd["bias_mus.2"].shape[0]
    hidden_dim = sd["weight_mus.0"].shape[0]
    num_layers = 2
    print(f"BNN from checkpoint: input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}")

    # Load train data and fit scalers (so dims match checkpoint: 840+25+140=1005)
    sys_train = load_and_reshape(os.path.join(data_dir, "sys_data_train.npy"))
    lat_train = load_and_reshape(os.path.join(data_dir, "lat_data_train.npy"))
    nxt_train = load_and_reshape(os.path.join(data_dir, "nxt_k_data_train.npy"))
    label_train = load_and_reshape(os.path.join(data_dir, "nxt_k_train_label.npy"))
    scaler_sys = StandardScaler().fit(sys_train)
    scaler_lat = StandardScaler().fit(lat_train)
    scaler_nxt = StandardScaler().fit(nxt_train)
    # Only first 2 outputs for pipeline (pred_lat, pred_viol)
    scaler_y = StandardScaler().fit(label_train[:, :2])

    sys_s = scaler_sys.transform(sys_train)
    lat_s = scaler_lat.transform(lat_train)
    nxt_s = scaler_nxt.transform(nxt_train)
    x_train = np.concatenate([sys_s, lat_s, nxt_s], axis=1)
    assert x_train.shape[1] == input_dim, f"x_train {x_train.shape[1]} vs input_dim {input_dim}"

    bnn = BayesianMLP(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    bnn.load_state_dict(sd, strict=True)
    bnn.eval()

    def process_split(suffix):
        sys_path = os.path.join(data_dir, f"sys_data_{suffix}.npy")
        if not os.path.isfile(sys_path):
            return None
        sys_data = load_and_reshape(sys_path)
        lat_data = load_and_reshape(os.path.join(data_dir, f"lat_data_{suffix}.npy"))
        nxt_data = load_and_reshape(os.path.join(data_dir, f"nxt_k_data_{suffix}.npy"))
        sys_s = scaler_sys.transform(sys_data)
        lat_s = scaler_lat.transform(lat_data)
        nxt_s = scaler_nxt.transform(nxt_data)
        return np.concatenate([sys_s, lat_s, nxt_s], axis=1)

    def run_bnn_mean(x_np, mc_samples, batch_size):
        n = x_np.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x_b = torch.tensor(x_np[start:end], dtype=torch.float32).to(device)
                batch_preds = []
                for _ in range(mc_samples):
                    pred = bnn(x_b, sample=True).cpu().numpy()
                    batch_preds.append(pred)
                batch_preds = np.stack(batch_preds, axis=0).mean(axis=0)
                all_preds.append(batch_preds)
        return np.concatenate(all_preds, axis=0)

    # Train (x_train already built above)
    print(f"Train X shape: {x_train.shape}")
    print(f"Running BNN with {args.mc_samples} MC samples...")
    y_bnn_train = run_bnn_mean(x_train, args.mc_samples, args.batch_size)
    y_bnn_train = y_bnn_train[:, :2]  # pred_lat, pred_viol
    y_bnn_train_real = scaler_y.inverse_transform(y_bnn_train)
    print(f"Y_bnn_train shape: {y_bnn_train_real.shape}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X_surrogate_train.npy"), x_train)
    np.save(os.path.join(args.out_dir, "Y_bnn_train.npy"), y_bnn_train_real)
    print(f"Saved X_surrogate_train.npy, Y_bnn_train.npy to {args.out_dir}")

    # Valid
    x_valid = process_split("valid")
    if x_valid is not None:
        y_bnn_valid = run_bnn_mean(x_valid, args.mc_samples, args.batch_size)
        y_bnn_valid = y_bnn_valid[:, :2]
        y_bnn_valid_real = scaler_y.inverse_transform(y_bnn_valid)
        np.save(os.path.join(args.out_dir, "X_surrogate_valid.npy"), x_valid)
        np.save(os.path.join(args.out_dir, "Y_bnn_valid.npy"), y_bnn_valid_real)
        print(f"Saved X_surrogate_valid.npy, Y_bnn_valid.npy (shape {y_bnn_valid_real.shape})")
    else:
        print("No valid split found; skipping.")
    if args.save_scalers:
        joblib.dump(scaler_sys, os.path.join(args.out_dir, "scaler_sys.pkl"))
        joblib.dump(scaler_lat, os.path.join(args.out_dir, "scaler_lat.pkl"))
        joblib.dump(scaler_nxt, os.path.join(args.out_dir, "scaler_nxt.pkl"))
        joblib.dump(scaler_y, os.path.join(args.out_dir, "scaler_y.pkl"))
        print("Saved scalers to out-dir (for predictor with full-dim surrogate).")
    print("Done.")


if __name__ == "__main__":
    main()
