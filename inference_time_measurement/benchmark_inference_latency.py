import argparse
import json
import os
import time

import joblib
import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICES = [
    "compose-post-redis",
    "compose-post-service",
    "home-timeline-redis",
    "home-timeline-service",
    "nginx-thrift",
    "post-storage-memcached",
    "post-storage-mongodb",
    "post-storage-service",
    "social-graph-mongodb",
    "social-graph-redis",
    "social-graph-service",
    "text-service",
    "text-filter-service",
    "unique-id-service",
    "url-shorten-service",
    "media-service",
    "media-filter-service",
    "user-mention-service",
    "user-memcached",
    "user-mongodb",
    "user-service",
    "user-timeline-mongodb",
    "user-timeline-redis",
    "user-timeline-service",
    "write-home-timeline-service",
    "write-home-timeline-rabbitmq",
    "write-user-timeline-service",
    "write-user-timeline-rabbitmq",
]
P99_SLICE = slice(15, 20)
P99_T1 = 15


class BayesianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.weight_mus = nn.ParameterList()
        self.bias_mus = nn.ParameterList()
        self.weight_logstds = nn.ParameterList()
        self.bias_logstds = nn.ParameterList()

        layer_dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            self.weight_mus.append(nn.Parameter(torch.randn(out_dim, in_dim) * 0.01))
            self.bias_mus.append(nn.Parameter(torch.zeros(out_dim)))
            self.weight_logstds.append(nn.Parameter(torch.ones(out_dim, in_dim) * -5))
            self.bias_logstds.append(nn.Parameter(torch.ones(out_dim) * -5))
        # Present in the original training/inference definition; keep for checkpoint compatibility.
        self.log_noise = nn.Parameter(torch.tensor(-3.0))

    def sample_weights(self):
        try:
            import pyro.distributions as dist
        except Exception:
            import torch.distributions as dist

        weights = []
        for w_mu, w_logstd, b_mu, b_logstd in zip(
            self.weight_mus, self.weight_logstds, self.bias_mus, self.bias_logstds
        ):
            w_std = torch.nn.functional.softplus(w_logstd)
            b_std = torch.nn.functional.softplus(b_logstd)
            w = dist.Normal(w_mu, w_std).rsample()
            b = dist.Normal(b_mu, b_std).rsample()
            weights.append((w, b))
        return weights

    def forward(self, x, sample=True):
        if sample:
            weights = self.sample_weights()
        else:
            weights = [(w_mu, b_mu) for w_mu, b_mu in zip(self.weight_mus, self.bias_mus)]
        for i, (w, b) in enumerate(weights):
            x = torch.nn.functional.linear(x, w, b)
            if i < len(weights) - 1:
                x = torch.relu(x)
        return x


def path_or_default(value, base_dir, default_leaf):
    if value is not None:
        return value
    return os.path.join(base_dir, default_leaf)


def compose_sys_data_channel(sys_data, field, batch_size, cnn_time_steps):
    for i, service in enumerate(SERVICES):
        assert len(sys_data[service][field]) == cnn_time_steps
        if i == 0:
            data = np.array(sys_data[service][field], dtype=np.float64)
        else:
            data = np.vstack((data, np.array(sys_data[service][field], dtype=np.float64)))

    data = data.reshape([1, data.shape[0], data.shape[1]])
    for i in range(0, batch_size):
        if i == 0:
            channel_data = np.array(data)
        else:
            channel_data = np.vstack((channel_data, data))
    channel_data = channel_data.reshape(
        [channel_data.shape[0], channel_data.shape[1] * channel_data.shape[2]]
    )
    return channel_data


def build_raw_features(info, nxt_horizon, cnn_time_steps):
    raw_sys_data = info["sys_data"]
    raw_next_info = info["next_info"]
    batch_size = len(raw_next_info)

    rps_data = compose_sys_data_channel(raw_sys_data, "rps", batch_size, cnn_time_steps)
    replica_data = compose_sys_data_channel(raw_sys_data, "replica", batch_size, cnn_time_steps)
    cpu_limit_data = compose_sys_data_channel(raw_sys_data, "cpu_limit", batch_size, cnn_time_steps)
    cpu_usage_mean_data = compose_sys_data_channel(raw_sys_data, "cpu_usage_mean", batch_size, cnn_time_steps)
    rss_mean_data = compose_sys_data_channel(raw_sys_data, "rss_mean", batch_size, cnn_time_steps)
    cache_mem_mean_data = compose_sys_data_channel(
        raw_sys_data, "cache_mem_mean", batch_size, cnn_time_steps
    )
    sys_data = np.concatenate(
        (
            rps_data,
            replica_data,
            cpu_limit_data,
            cpu_usage_mean_data,
            rss_mean_data,
            cache_mem_mean_data,
        ),
        axis=1,
    )

    for key in ["90.0", "95.0", "98.0", "99.0", "99.9"]:
        assert len(raw_sys_data["e2e_lat"][key]) == cnn_time_steps
        if key == "90.0":
            e2e_lat = np.array(raw_sys_data["e2e_lat"][key], dtype=np.float64)
        else:
            e2e_lat = np.vstack((e2e_lat, np.array(raw_sys_data["e2e_lat"][key], dtype=np.float64)))
    e2e_lat = e2e_lat.reshape([1, e2e_lat.shape[0], e2e_lat.shape[1]])
    lat_data = np.repeat(e2e_lat, batch_size, axis=0).reshape(batch_size, -1)

    nxt_rows = []
    for proposal in raw_next_info:
        ncore = np.array([proposal[s]["cpus"] for s in SERVICES], dtype=np.float64)
        nxt_mat = np.tile(ncore.reshape(-1, 1), (1, nxt_horizon))
        nxt_rows.append(nxt_mat.reshape(-1))
    nxt_data = np.vstack(nxt_rows)

    return sys_data, lat_data, nxt_data


def prepare_x(sys_data, lat_data, nxt_data, scaler_sys, scaler_lat, scaler_nxt, top_indices):
    sys_scaled = scaler_sys.transform(sys_data)
    lat_scaled = scaler_lat.transform(lat_data)
    nxt_scaled = scaler_nxt.transform(nxt_data)
    x = np.concatenate([sys_scaled, lat_scaled, nxt_scaled], axis=1)
    return x[:, top_indices]


def weighted_quantile(values, weights, q):
    if len(values) == 0:
        return 0.0
    q = float(np.clip(q, 0.0, 1.0))
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return float(np.quantile(values, q))
    cdf = np.cumsum(weights) / weight_sum
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def distance(x, y, metric):
    diff = x - y
    if metric == "l1":
        return np.sum(np.abs(diff), axis=1)
    return np.linalg.norm(diff, axis=1)


class LatencyBench:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.surrogate_model = None
        self.sur_scaler_sys = None
        self.sur_scaler_lat = None
        self.sur_scaler_nxt = None
        self.sur_top_indices = None
        self.sur_nxt_horizon = None

        self.bnn = None
        self.bnn_scaler_sys = None
        self.bnn_scaler_lat = None
        self.bnn_scaler_nxt = None
        self.bnn_scaler_y = None
        self.bnn_top_indices = None
        self.bnn_nxt_horizon = None

        self.cp_x_cal = None
        self.cp_scores = None
        self.cp_enabled = False

    def load_surrogate(self):
        mdir = self.args.surrogate_model_dir
        model_path = self.args.surrogate_model_path or os.path.join(mdir, "bnn_surrogate_tree.joblib")
        sys_path = path_or_default(self.args.surrogate_scaler_sys, mdir, "scaler_sys.pkl")
        lat_path = path_or_default(self.args.surrogate_scaler_lat, mdir, "scaler_lat.pkl")
        nxt_path = path_or_default(self.args.surrogate_scaler_nxt, mdir, "scaler_nxt.pkl")
        top_path = path_or_default(self.args.surrogate_top_features, mdir, "top_feature_indices.npy")

        self.sur_scaler_sys = joblib.load(sys_path)
        self.sur_scaler_lat = joblib.load(lat_path)
        self.sur_scaler_nxt = joblib.load(nxt_path)
        self.sur_top_indices = np.load(top_path)
        self.surrogate_model = joblib.load(model_path)

        n_nxt = int(self.sur_scaler_nxt.n_features_in_)
        if n_nxt % len(SERVICES) != 0:
            raise ValueError("Surrogate scaler_nxt shape mismatch with service count")
        self.sur_nxt_horizon = n_nxt // len(SERVICES)

    def load_bnn(self):
        mdir = self.args.bnn_model_dir
        prefix = self.args.bnn_artifact_prefix
        model_path = self.args.bnn_model_path or os.path.join(mdir, f"{prefix}_model.pth")
        top_path = self.args.bnn_top_features or os.path.join(mdir, f"{prefix}_top_indices.npy")
        scalers_bundle = self.args.bnn_scalers_bundle or os.path.join(mdir, f"{prefix}_scalers.pkl")

        if os.path.isfile(scalers_bundle):
            (
                self.bnn_scaler_sys,
                self.bnn_scaler_lat,
                self.bnn_scaler_nxt,
                self.bnn_scaler_y,
            ) = joblib.load(scalers_bundle)
        else:
            self.bnn_scaler_sys = joblib.load(path_or_default(self.args.bnn_scaler_sys, mdir, "scaler_sys.pkl"))
            self.bnn_scaler_lat = joblib.load(path_or_default(self.args.bnn_scaler_lat, mdir, "scaler_lat.pkl"))
            self.bnn_scaler_nxt = joblib.load(path_or_default(self.args.bnn_scaler_nxt, mdir, "scaler_nxt.pkl"))
            self.bnn_scaler_y = joblib.load(path_or_default(self.args.bnn_scaler_y, mdir, "scaler_y.pkl"))

        if os.path.isfile(top_path):
            self.bnn_top_indices = np.load(top_path)
        else:
            self.bnn_top_indices = np.load(os.path.join(mdir, "top_feature_indices.npy"))

        n_nxt = int(self.bnn_scaler_nxt.n_features_in_)
        if n_nxt % len(SERVICES) != 0:
            raise ValueError("BNN scaler_nxt shape mismatch with service count")
        self.bnn_nxt_horizon = n_nxt // len(SERVICES)

        input_dim = len(self.bnn_top_indices)
        output_dim = len(self.bnn_scaler_y.mean_)
        self.bnn = BayesianMLP(input_dim, output_dim, self.args.bnn_hidden_dim, self.args.bnn_num_layers).to(
            self.device
        )
        state = torch.load(model_path, map_location=self.device)
        self.bnn.load_state_dict(state)
        self.bnn.eval()

    def load_calibration(self):
        if self.args.calibration_data is None:
            self.cp_enabled = False
            return
        blob = np.load(self.args.calibration_data)
        if "X_cal" not in blob or "Y_cal" not in blob:
            raise ValueError("Calibration .npz must contain arrays X_cal and Y_cal")

        self.cp_x_cal = np.asarray(blob["X_cal"], dtype=np.float64)
        y_cal = np.asarray(blob["Y_cal"], dtype=np.float64)
        if y_cal.ndim == 1:
            y_cal = y_cal.reshape(-1, 1)
        if y_cal.shape[1] < 2:
            raise ValueError("Y_cal must have at least 2 columns: [latency, viol_prob]")

        pred_cal = self.surrogate_model.predict(self.cp_x_cal)
        if pred_cal.ndim == 1:
            pred_cal = pred_cal.reshape(-1, 1)
        if pred_cal.shape[1] == 1:
            pred_cal = np.hstack([pred_cal, pred_cal])
        pred_cal = pred_cal[:, :2]
        self.cp_scores = np.max(np.abs(y_cal[:, :2] - pred_cal), axis=1)
        self.cp_enabled = True

    def surrogate_predict(self, info):
        sys_sur, lat_sur, nxt_sur = build_raw_features(info, self.sur_nxt_horizon, self.args.cnn_time_steps)
        x_sur = prepare_x(
            sys_sur,
            lat_sur,
            nxt_sur,
            self.sur_scaler_sys,
            self.sur_scaler_lat,
            self.sur_scaler_nxt,
            self.sur_top_indices,
        )
        pred = self.surrogate_model.predict(x_sur)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if pred.shape[1] == 1:
            pred = np.hstack([pred, pred])
        return pred[:, :2], x_sur

    def surrogate_prepare_x(self, info):
        sys_sur, lat_sur, nxt_sur = build_raw_features(info, self.sur_nxt_horizon, self.args.cnn_time_steps)
        return prepare_x(
            sys_sur,
            lat_sur,
            nxt_sur,
            self.sur_scaler_sys,
            self.sur_scaler_lat,
            self.sur_scaler_nxt,
            self.sur_top_indices,
        )

    def conformal_only_from_x(self, x_sur):
        """
        Conformal overhead only: compute widths + uncertain mask from x_sur.
        Does NOT include surrogate predict or any BNN fallback.
        """
        if not self.cp_enabled:
            return None
        widths = np.zeros((x_sur.shape[0], 2), dtype=np.float64)
        for i in range(x_sur.shape[0]):
            d = distance(self.cp_x_cal, x_sur[i : i + 1], self.args.cp_distance)
            tau = max(float(self.args.cp_tau), 1e-12)
            w = np.exp(-d / tau)
            q = weighted_quantile(self.cp_scores, w, 1.0 - float(self.args.alpha))
            widths[i, :] = q
        uncertain = (2.0 * widths[:, 0] > self.args.cp_max_width_latency) | (
            2.0 * widths[:, 1] > self.args.cp_max_width_viol
        )
        return widths, uncertain

    def bnn_predict(self, info):
        sys_bnn, lat_bnn, nxt_bnn = build_raw_features(info, self.bnn_nxt_horizon, self.args.cnn_time_steps)
        x_bnn = prepare_x(
            sys_bnn,
            lat_bnn,
            nxt_bnn,
            self.bnn_scaler_sys,
            self.bnn_scaler_lat,
            self.bnn_scaler_nxt,
            self.bnn_top_indices,
        )
        return self.bnn_predict_from_x(x_bnn)

    def bnn_prepare_x(self, info):
        sys_bnn, lat_bnn, nxt_bnn = build_raw_features(info, self.bnn_nxt_horizon, self.args.cnn_time_steps)
        return prepare_x(
            sys_bnn,
            lat_bnn,
            nxt_bnn,
            self.bnn_scaler_sys,
            self.bnn_scaler_lat,
            self.bnn_scaler_nxt,
            self.bnn_top_indices,
        )

    def bnn_predict_from_x(self, x_bnn):
        if x_bnn.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        x_tensor = torch.tensor(x_bnn, dtype=torch.float32, device=self.device)
        mc_preds = []
        with torch.no_grad():
            for _ in range(self.args.bnn_mc_samples):
                mc_preds.append(self.bnn(x_tensor, sample=True).cpu().numpy())
        mc_preds = np.stack(mc_preds)
        m_samples, batch_size, out_dim = mc_preds.shape

        flat = mc_preds.reshape(-1, out_dim)
        pred_real_mc = self.bnn_scaler_y.inverse_transform(flat).reshape(m_samples, batch_size, out_dim)
        if out_dim > P99_T1:
            p99_t1 = pred_real_mc[:, :, P99_T1].mean(axis=0)
            p99_end = min(out_dim, P99_SLICE.stop)
            if p99_end > P99_T1:
                p99_horizon = pred_real_mc[:, :, P99_T1:p99_end]
                viol_prob = (np.max(p99_horizon, axis=2) >= float(self.args.qos)).mean(axis=0)
            else:
                viol_prob = (pred_real_mc[:, :, P99_T1] >= float(self.args.qos)).mean(axis=0)
        else:
            p99_t1 = pred_real_mc[:, :, 0].mean(axis=0)
            viol_prob = np.zeros(batch_size, dtype=np.float64)
        return np.column_stack([p99_t1, viol_prob])

    def conformal_predict(self, info):
        sur_pred, x_sur = self.surrogate_predict(info)
        result = np.array(sur_pred, dtype=np.float64)

        if not self.cp_enabled:
            bnn_indices = np.arange(result.shape[0])
        else:
            widths = np.zeros((x_sur.shape[0], 2), dtype=np.float64)
            for i in range(x_sur.shape[0]):
                d = distance(self.cp_x_cal, x_sur[i : i + 1], self.args.cp_distance)
                tau = max(float(self.args.cp_tau), 1e-12)
                w = np.exp(-d / tau)
                q = weighted_quantile(self.cp_scores, w, 1.0 - float(self.args.alpha))
                widths[i, :] = q
            uncertain = (2.0 * widths[:, 0] > self.args.cp_max_width_latency) | (
                2.0 * widths[:, 1] > self.args.cp_max_width_viol
            )
            bnn_indices = np.where(uncertain)[0]

        if len(bnn_indices) > 0:
            sub_info = {"sys_data": info["sys_data"], "next_info": [info["next_info"][i] for i in bnn_indices]}
            sys_bnn, lat_bnn, nxt_bnn = build_raw_features(
                sub_info, self.bnn_nxt_horizon, self.args.cnn_time_steps
            )
            x_bnn = prepare_x(
                sys_bnn,
                lat_bnn,
                nxt_bnn,
                self.bnn_scaler_sys,
                self.bnn_scaler_lat,
                self.bnn_scaler_nxt,
                self.bnn_top_indices,
            )
            bnn_pred = self.bnn_predict_from_x(x_bnn)
            for j, idx in enumerate(bnn_indices):
                result[idx, 0] = bnn_pred[j, 0]
                result[idx, 1] = bnn_pred[j, 1]
        return result

    def conformal_predict_from_x(self, x_sur, x_bnn):
        """
        Conformal pipeline without preprocessing:
        - surrogate model inference on cached x_sur
        - conformal uncertainty scoring
        - BNN fallback inference on cached x_bnn subset
        """
        pred = self.surrogate_model.predict(x_sur)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if pred.shape[1] == 1:
            pred = np.hstack([pred, pred])
        result = np.array(pred[:, :2], dtype=np.float64)

        if not self.cp_enabled:
            bnn_indices = np.arange(result.shape[0])
        else:
            widths = np.zeros((x_sur.shape[0], 2), dtype=np.float64)
            for i in range(x_sur.shape[0]):
                d = distance(self.cp_x_cal, x_sur[i : i + 1], self.args.cp_distance)
                tau = max(float(self.args.cp_tau), 1e-12)
                w = np.exp(-d / tau)
                q = weighted_quantile(self.cp_scores, w, 1.0 - float(self.args.alpha))
                widths[i, :] = q
            uncertain = (2.0 * widths[:, 0] > self.args.cp_max_width_latency) | (
                2.0 * widths[:, 1] > self.args.cp_max_width_viol
            )
            bnn_indices = np.where(uncertain)[0]

        if len(bnn_indices) > 0:
            bnn_pred = self.bnn_predict_from_x(x_bnn[bnn_indices])
            result[bnn_indices, 0] = bnn_pred[:, 0]
            result[bnn_indices, 1] = bnn_pred[:, 1]
        return result


def generate_synthetic_info(batch_size, cnn_time_steps, seed):
    rng = np.random.default_rng(seed)
    sys_data = {}
    fields = ["rps", "replica", "cpu_limit", "cpu_usage_mean", "rss_mean", "cache_mem_mean"]
    for service in SERVICES:
        sys_data[service] = {}
        for field in fields:
            if field == "replica":
                vals = rng.integers(1, 5, size=cnn_time_steps).astype(float)
            elif field == "cpu_usage_mean":
                vals = rng.uniform(0.1, 0.95, size=cnn_time_steps)
            elif field == "cpu_limit":
                vals = rng.uniform(0.3, 2.0, size=cnn_time_steps)
            elif field in ("rss_mean", "cache_mem_mean"):
                vals = rng.uniform(50.0, 800.0, size=cnn_time_steps)
            else:
                vals = rng.uniform(50.0, 1500.0, size=cnn_time_steps)
            sys_data[service][field] = vals.tolist()

    sys_data["e2e_lat"] = {
        "90.0": rng.uniform(60.0, 260.0, size=cnn_time_steps).tolist(),
        "95.0": rng.uniform(80.0, 320.0, size=cnn_time_steps).tolist(),
        "98.0": rng.uniform(120.0, 420.0, size=cnn_time_steps).tolist(),
        "99.0": rng.uniform(160.0, 550.0, size=cnn_time_steps).tolist(),
        "99.9": rng.uniform(220.0, 800.0, size=cnn_time_steps).tolist(),
    }

    next_info = []
    for _ in range(batch_size):
        proposal = {}
        for service in SERVICES:
            proposal[service] = {"cpus": float(rng.uniform(0.2, 4.0))}
        next_info.append(proposal)

    return {"sys_data": sys_data, "next_info": next_info}


def load_input_info(input_json, batch_size, cnn_time_steps, seed):
    if input_json is None:
        return generate_synthetic_info(batch_size=batch_size, cnn_time_steps=cnn_time_steps, seed=seed)
    with open(input_json, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if "sys_data" not in obj or "next_info" not in obj:
        raise ValueError("Input JSON must contain 'sys_data' and 'next_info'")
    return obj


def l2_distance_uncertainty_step(v_a, v_b, threshold):
    """
    Minimal distance-based uncertainty step: L2 distance between two vectors that have the
    same dimension as model inputs, then compare to a constant threshold.
    """
    diff = v_a - v_b
    d = float(np.sqrt(np.dot(diff, diff)))
    return d > threshold


def l2_batch_distance_uncertainty_vs_ref(x, ref, threshold):
    """
    For each row, L2(x[i], ref) and compare to threshold (batch of independent pair checks).
    """
    uncertain = np.zeros(x.shape[0], dtype=bool)
    for i in range(x.shape[0]):
        diff = x[i] - ref
        d = float(np.sqrt(np.dot(diff, diff)))
        uncertain[i] = d > threshold
    return uncertain


def measure_latency(callable_fn, warmup, runs):
    for _ in range(warmup):
        _ = callable_fn()

    lat_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = callable_fn()
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
    lat_ms = np.asarray(lat_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(lat_ms)),
        "p50_ms": float(np.percentile(lat_ms, 50)),
        "p95_ms": float(np.percentile(lat_ms, 95)),
        "p99_ms": float(np.percentile(lat_ms, 99)),
        "min_ms": float(np.min(lat_ms)),
        "max_ms": float(np.max(lat_ms)),
    }


def print_stats(name, stats):
    print(
        f"{name:24s} mean={stats['mean_ms']:.3f}ms "
        f"p50={stats['p50_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms p99={stats['p99_ms']:.3f}ms "
        f"min={stats['min_ms']:.3f}ms max={stats['max_ms']:.3f}ms"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure latency for surrogate, BNN, and conformal pipeline."
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cnn-time-steps", type=int, default=5)
    parser.add_argument("--input-json", type=str, default=None)

    parser.add_argument(
        "--surrogate-model-dir",
        type=str,
        default=os.path.normpath(os.path.join(SCRIPT_DIR, "surrogate_dt_model")),
    )
    parser.add_argument("--surrogate-model-path", type=str, default=None)
    parser.add_argument("--surrogate-scaler-sys", type=str, default=None)
    parser.add_argument("--surrogate-scaler-lat", type=str, default=None)
    parser.add_argument("--surrogate-scaler-nxt", type=str, default=None)
    parser.add_argument("--surrogate-top-features", type=str, default=None)

    parser.add_argument("--bnn-model-dir", type=str, default=os.path.join(SCRIPT_DIR, "bnn_model"))
    parser.add_argument("--bnn-artifact-prefix", type=str, default="bnn_layers2_hdim700_lr1e-04")
    parser.add_argument("--bnn-model-path", type=str, default=None)
    parser.add_argument("--bnn-scalers-bundle", type=str, default=None)
    parser.add_argument("--bnn-scaler-sys", type=str, default=None)
    parser.add_argument("--bnn-scaler-lat", type=str, default=None)
    parser.add_argument("--bnn-scaler-nxt", type=str, default=None)
    parser.add_argument("--bnn-scaler-y", type=str, default=None)
    parser.add_argument("--bnn-top-features", type=str, default=None)
    parser.add_argument("--bnn-num-layers", type=int, default=2)
    parser.add_argument("--bnn-hidden-dim", type=int, default=700)
    parser.add_argument("--bnn-mc-samples", type=int, default=50)
    parser.add_argument("--qos", type=float, default=500.0)

    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cp-tau", type=float, default=1.0)
    parser.add_argument("--cp-distance", type=str, default="l2", choices=["l1", "l2"])
    parser.add_argument("--cp-max-width-latency", type=float, default=120.0)
    parser.add_argument("--cp-max-width-viol", type=float, default=0.25)
    parser.add_argument(
        "--l2-uncertainty-threshold",
        type=float,
        default=1.0,
        help="Distance threshold for l2_pair_uncertainty_micro / l2_batch_vs_ref_uncertainty_micro.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bench = LatencyBench(args)
    bench.load_surrogate()
    bench.load_bnn()
    bench.load_calibration()

    info = load_input_info(args.input_json, args.batch_size, args.cnn_time_steps, args.seed)
    batch_size = len(info["next_info"])

    x_sur_cached = bench.surrogate_prepare_x(info)
    x_bnn_cached = bench.bnn_prepare_x(info)

    v_a = np.asarray(x_sur_cached[0], dtype=np.float64)
    if batch_size > 1:
        v_b = np.asarray(x_sur_cached[1], dtype=np.float64)
    elif bench.cp_enabled and bench.cp_x_cal is not None and len(bench.cp_x_cal) > 0:
        v_b = np.asarray(bench.cp_x_cal[0], dtype=np.float64)
    else:
        v_b = v_a + 1e-6
    ref_l2 = (
        np.asarray(bench.cp_x_cal[0], dtype=np.float64)
        if bench.cp_enabled and bench.cp_x_cal is not None and len(bench.cp_x_cal) > 0
        else v_a.copy()
    )

    l2_pair_micro_stats = measure_latency(
        callable_fn=lambda: l2_distance_uncertainty_step(v_a, v_b, args.l2_uncertainty_threshold),
        warmup=args.warmup,
        runs=args.runs,
    )
    l2_batch_vs_ref_stats = measure_latency(
        callable_fn=lambda: l2_batch_distance_uncertainty_vs_ref(
            x_sur_cached.astype(np.float64, copy=False),
            ref_l2,
            args.l2_uncertainty_threshold,
        ),
        warmup=args.warmup,
        runs=args.runs,
    )

    surrogate_stats = measure_latency(
        callable_fn=lambda: bench.surrogate_predict(info)[0], warmup=args.warmup, runs=args.runs
    )
    surrogate_model_only_stats = measure_latency(
        callable_fn=lambda: bench.surrogate_model.predict(x_sur_cached),
        warmup=args.warmup,
        runs=args.runs,
    )
    bnn_stats = measure_latency(
        callable_fn=lambda: bench.bnn_predict(info), warmup=args.warmup, runs=args.runs
    )
    bnn_model_only_stats = measure_latency(
        callable_fn=lambda: bench.bnn_predict_from_x(x_bnn_cached),
        warmup=args.warmup,
        runs=args.runs,
    )
    cp_only_stats = None
    if bench.cp_enabled:
        cp_only_stats = measure_latency(
            callable_fn=lambda: bench.conformal_only_from_x(x_sur_cached),
            warmup=args.warmup,
            runs=args.runs,
        )
    conformal_stats = measure_latency(
        callable_fn=lambda: bench.conformal_predict_from_x(x_sur_cached, x_bnn_cached),
        warmup=args.warmup,
        runs=args.runs,
    )

    print(f"Input batch size: {batch_size}")
    print(f"Warmup runs: {args.warmup}, measured runs: {args.runs}")
    print_stats("surrogate_only", surrogate_stats)
    print_stats("surrogate_model_only", surrogate_model_only_stats)
    print_stats("bnn_only", bnn_stats)
    print_stats("bnn_model_only", bnn_model_only_stats)
    print_stats("l2_pair_uncertainty_micro", l2_pair_micro_stats)
    print_stats("l2_batch_vs_ref_uncertainty_micro", l2_batch_vs_ref_stats)
    if cp_only_stats is None:
        print(f"{'conformal_only':24s} N/A (calibration disabled; pass --calibration-data)")
    else:
        print_stats("conformal_only", cp_only_stats)
    print_stats("conformal_pipeline_model_only", conformal_stats)


if __name__ == "__main__":
    main()
