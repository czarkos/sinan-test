import os
import socket
import json
import argparse
import logging

import joblib
import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

Services = [
    'compose-post-redis',
    'compose-post-service',
    'home-timeline-redis',
    'home-timeline-service',
    'nginx-thrift',
    'post-storage-memcached',
    'post-storage-mongodb',
    'post-storage-service',
    'social-graph-mongodb',
    'social-graph-redis',
    'social-graph-service',
    'text-service',
    'text-filter-service',
    'unique-id-service',
    'url-shorten-service',
    'media-service',
    'media-filter-service',
    'user-mention-service',
    'user-memcached',
    'user-mongodb',
    'user-service',
    'user-timeline-mongodb',
    'user-timeline-redis',
    'user-timeline-service',
    'write-home-timeline-service',
    'write-home-timeline-rabbitmq',
    'write-user-timeline-service',
    'write-user-timeline-rabbitmq',
]


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

    def sample_weights(self):
        import pyro.distributions as dist

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


parser = argparse.ArgumentParser()
parser.add_argument('--server-port', dest='server_port', type=int, default=40010)
parser.add_argument('--cnn-time-steps', dest='cnn_time_steps', type=int, default=5)

# Surrogate artifacts
parser.add_argument(
    '--surrogate-model-dir',
    type=str,
    default=os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'surrogate', 'model')),
)
parser.add_argument('--surrogate-model-path', type=str, default=None)
parser.add_argument('--surrogate-scaler-sys', type=str, default=None)
parser.add_argument('--surrogate-scaler-lat', type=str, default=None)
parser.add_argument('--surrogate-scaler-nxt', type=str, default=None)
parser.add_argument('--surrogate-scaler-y', type=str, default=None)
parser.add_argument('--surrogate-top-features', type=str, default=None)

# BNN artifacts
parser.add_argument(
    '--bnn-model-dir',
    type=str,
    default=os.path.join(_SCRIPT_DIR, 'model'),
)
parser.add_argument('--bnn-artifact-prefix', type=str, default='bnn_layers2_hdim700_lr1e-04')
parser.add_argument('--bnn-model-path', type=str, default=None)
parser.add_argument('--bnn-scalers-bundle', type=str, default=None)
parser.add_argument('--bnn-scaler-sys', type=str, default=None)
parser.add_argument('--bnn-scaler-lat', type=str, default=None)
parser.add_argument('--bnn-scaler-nxt', type=str, default=None)
parser.add_argument('--bnn-scaler-y', type=str, default=None)
parser.add_argument('--bnn-top-features', type=str, default=None)
parser.add_argument('--bnn-num-layers', type=int, default=2)
parser.add_argument('--bnn-hidden-dim', type=int, default=700)
parser.add_argument('--bnn-mc-samples', type=int, default=50)
parser.add_argument('--qos', type=float, default=500.0)

# Conformal uncertainty
parser.add_argument(
    '--calibration-data',
    type=str,
    default=os.path.normpath(
        os.path.join(_SCRIPT_DIR, '..', 'surrogate', 'data', 'surrogate_cp_calibration.npz')
    ),
    help='Path to .npz with arrays X_cal (selected surrogate features) and Y_cal (2 targets)',
)
parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level (1-alpha coverage)')
parser.add_argument(
    '--cp-tau',
    type=float,
    default=1.0,
    help='Temperature for distance-based weights exp(-distance/tau)',
)
parser.add_argument(
    '--cp-distance',
    type=str,
    default='l2',
    choices=['l2', 'l1'],
    help='Distance metric for weighted conformal prediction',
)
parser.add_argument(
    '--cp-max-width-latency',
    type=float,
    default=120.0,
    help='Fallback to BNN if surrogate CP interval width for latency exceeds this',
)
parser.add_argument(
    '--cp-max-width-viol',
    type=float,
    default=0.25,
    help='Fallback to BNN if surrogate CP interval width for violation prob exceeds this',
)

args = parser.parse_args()

ServerPort = args.server_port
CnnTimeSteps = args.cnn_time_steps
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_P99_SLICE = slice(15, 20)
_P99_T1 = 15

surrogate_model = None
sur_scaler_sys = None
sur_scaler_lat = None
sur_scaler_nxt = None
sur_scaler_y = None
sur_top_indices = None
sur_nxt_horizon = None

bnn = None
bnn_scaler_sys = None
bnn_scaler_lat = None
bnn_scaler_nxt = None
bnn_scaler_y = None
bnn_top_indices = None
bnn_nxt_horizon = None

cp_x_cal = None
cp_scores = None
cp_enabled = False


def _path_or_default(value, base_dir, default_leaf):
    if value is not None:
        return value
    return os.path.join(base_dir, default_leaf)


def _compose_sys_data_channel(sys_data, field, batch_size):
    for i, service in enumerate(Services):
        assert len(sys_data[service][field]) == CnnTimeSteps
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


def _build_raw_features(info, nxt_horizon):
    raw_sys_data = info['sys_data']
    raw_next_info = info['next_info']
    batch_size = len(raw_next_info)

    rps_data = _compose_sys_data_channel(raw_sys_data, 'rps', batch_size)
    replica_data = _compose_sys_data_channel(raw_sys_data, 'replica', batch_size)
    cpu_limit_data = _compose_sys_data_channel(raw_sys_data, 'cpu_limit', batch_size)
    cpu_usage_mean_data = _compose_sys_data_channel(raw_sys_data, 'cpu_usage_mean', batch_size)
    rss_mean_data = _compose_sys_data_channel(raw_sys_data, 'rss_mean', batch_size)
    cache_mem_mean_data = _compose_sys_data_channel(raw_sys_data, 'cache_mem_mean', batch_size)

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

    for key in ['90.0', '95.0', '98.0', '99.0', '99.9']:
        assert len(raw_sys_data['e2e_lat'][key]) == CnnTimeSteps
        if key == '90.0':
            e2e_lat = np.array(raw_sys_data['e2e_lat'][key], dtype=np.float64)
        else:
            e2e_lat = np.vstack((e2e_lat, np.array(raw_sys_data['e2e_lat'][key], dtype=np.float64)))
    e2e_lat = e2e_lat.reshape([1, e2e_lat.shape[0], e2e_lat.shape[1]])
    lat_data = np.repeat(e2e_lat, batch_size, axis=0).reshape(batch_size, -1)

    nxt_rows = []
    for proposal in raw_next_info:
        ncore = np.array([proposal[s]['cpus'] for s in Services], dtype=np.float64)
        nxt_mat = np.tile(ncore.reshape(-1, 1), (1, nxt_horizon))
        nxt_rows.append(nxt_mat.reshape(-1))
    nxt_data = np.vstack(nxt_rows)
    return sys_data, lat_data, nxt_data


def _prepare_x(sys_data, lat_data, nxt_data, scaler_sys, scaler_lat, scaler_nxt, top_indices):
    sys_scaled = scaler_sys.transform(sys_data)
    lat_scaled = scaler_lat.transform(lat_data)
    nxt_scaled = scaler_nxt.transform(nxt_data)
    x = np.concatenate([sys_scaled, lat_scaled, nxt_scaled], axis=1)
    return x[:, top_indices]


def _weighted_quantile(values, weights, q):
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
    idx = int(np.searchsorted(cdf, q, side='left'))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def _distance(x, y):
    diff = x - y
    if args.cp_distance == 'l1':
        return np.sum(np.abs(diff), axis=1)
    return np.linalg.norm(diff, axis=1)


def _load_surrogate():
    global surrogate_model, sur_scaler_sys, sur_scaler_lat, sur_scaler_nxt, sur_scaler_y
    global sur_top_indices, sur_nxt_horizon
    mdir = args.surrogate_model_dir
    model_path = args.surrogate_model_path or os.path.join(mdir, 'bnn_surrogate_tree.joblib')
    sys_path = _path_or_default(args.surrogate_scaler_sys, mdir, 'scaler_sys.pkl')
    lat_path = _path_or_default(args.surrogate_scaler_lat, mdir, 'scaler_lat.pkl')
    nxt_path = _path_or_default(args.surrogate_scaler_nxt, mdir, 'scaler_nxt.pkl')
    y_path = _path_or_default(args.surrogate_scaler_y, mdir, 'scaler_y.pkl')
    top_path = _path_or_default(args.surrogate_top_features, mdir, 'top_feature_indices.npy')

    sur_scaler_sys = joblib.load(sys_path)
    sur_scaler_lat = joblib.load(lat_path)
    sur_scaler_nxt = joblib.load(nxt_path)
    sur_scaler_y = joblib.load(y_path)
    sur_top_indices = np.load(top_path)
    surrogate_model = joblib.load(model_path)

    n_nxt = int(sur_scaler_nxt.n_features_in_)
    if n_nxt % len(Services) != 0:
        raise ValueError('Surrogate scaler_nxt shape mismatch with service count')
    sur_nxt_horizon = n_nxt // len(Services)
    logging.info('Loaded surrogate artifacts from %s', mdir)


def _load_bnn():
    global bnn, bnn_scaler_sys, bnn_scaler_lat, bnn_scaler_nxt, bnn_scaler_y
    global bnn_top_indices, bnn_nxt_horizon
    mdir = args.bnn_model_dir
    prefix = args.bnn_artifact_prefix
    model_path = args.bnn_model_path or os.path.join(mdir, '%s_model.pth' % prefix)
    top_path = args.bnn_top_features or os.path.join(mdir, '%s_top_indices.npy' % prefix)
    scalers_bundle = args.bnn_scalers_bundle or os.path.join(mdir, '%s_scalers.pkl' % prefix)

    if os.path.isfile(scalers_bundle):
        bnn_scaler_sys, bnn_scaler_lat, bnn_scaler_nxt, bnn_scaler_y = joblib.load(scalers_bundle)
    else:
        bnn_scaler_sys = joblib.load(_path_or_default(args.bnn_scaler_sys, mdir, 'scaler_sys.pkl'))
        bnn_scaler_lat = joblib.load(_path_or_default(args.bnn_scaler_lat, mdir, 'scaler_lat.pkl'))
        bnn_scaler_nxt = joblib.load(_path_or_default(args.bnn_scaler_nxt, mdir, 'scaler_nxt.pkl'))
        bnn_scaler_y = joblib.load(_path_or_default(args.bnn_scaler_y, mdir, 'scaler_y.pkl'))

    if os.path.isfile(top_path):
        bnn_top_indices = np.load(top_path)
    else:
        alt_top = os.path.join(mdir, 'top_feature_indices.npy')
        bnn_top_indices = np.load(alt_top)

    n_nxt = int(bnn_scaler_nxt.n_features_in_)
    if n_nxt % len(Services) != 0:
        raise ValueError('BNN scaler_nxt shape mismatch with service count')
    bnn_nxt_horizon = n_nxt // len(Services)

    input_dim = len(bnn_top_indices)
    output_dim = len(bnn_scaler_y.mean_)
    bnn = BayesianMLP(input_dim, output_dim, args.bnn_hidden_dim, args.bnn_num_layers).to(device)
    state = torch.load(model_path, map_location=device)
    bnn.load_state_dict(state)
    bnn.eval()
    logging.info('Loaded BNN artifacts from %s', mdir)


def _load_calibration():
    global cp_x_cal, cp_scores, cp_enabled
    if args.calibration_data is None:
        cp_enabled = False
        logging.warning('No --calibration-data provided; fallback-to-BNN mode will be used')
        return

    blob = np.load(args.calibration_data)
    if 'X_cal' not in blob or 'Y_cal' not in blob:
        raise ValueError('Calibration .npz must contain arrays X_cal and Y_cal')

    cp_x_cal = np.asarray(blob['X_cal'], dtype=np.float64)
    y_cal = np.asarray(blob['Y_cal'], dtype=np.float64)
    if y_cal.ndim == 1:
        y_cal = y_cal.reshape(-1, 1)
    if y_cal.shape[1] < 2:
        raise ValueError('Y_cal must have at least 2 columns: [latency, viol_prob]')

    pred_cal = surrogate_model.predict(cp_x_cal)
    if pred_cal.ndim == 1:
        pred_cal = pred_cal.reshape(-1, 1)
    if pred_cal.shape[1] == 1:
        pred_cal = np.hstack([pred_cal, pred_cal])
    pred_cal = pred_cal[:, :2]
    cp_scores = np.max(np.abs(y_cal[:, :2] - pred_cal), axis=1)
    cp_enabled = True
    logging.info('Loaded calibration set: n=%d, alpha=%.3f, tau=%.3f', len(cp_scores), args.alpha, args.cp_tau)


def _surrogate_predict_with_cp(x_sur):
    pred = surrogate_model.predict(x_sur)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if pred.shape[1] == 1:
        pred = np.hstack([pred, pred])
    pred = pred[:, :2]

    widths = np.zeros((x_sur.shape[0], 2), dtype=np.float64)
    uncertain = np.ones(x_sur.shape[0], dtype=bool)
    if cp_enabled:
        for i in range(x_sur.shape[0]):
            d = _distance(cp_x_cal, x_sur[i : i + 1])
            tau = max(float(args.cp_tau), 1e-12)
            w = np.exp(-d / tau)
            q = _weighted_quantile(cp_scores, w, 1.0 - float(args.alpha))
            widths[i, :] = q
        uncertain = (2.0 * widths[:, 0] > args.cp_max_width_latency) | (
            2.0 * widths[:, 1] > args.cp_max_width_viol
        )
    return pred, widths, uncertain


def _bnn_predict_from_x(x_bnn):
    if x_bnn.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    x_tensor = torch.tensor(x_bnn, dtype=torch.float32, device=device)
    mc_preds = []
    with torch.no_grad():
        for _ in range(args.bnn_mc_samples):
            mc_preds.append(bnn(x_tensor, sample=True).cpu().numpy())
    mc_preds = np.stack(mc_preds)
    m_samples, batch_size, out_dim = mc_preds.shape

    flat = mc_preds.reshape(-1, out_dim)
    pred_real_mc = bnn_scaler_y.inverse_transform(flat).reshape(m_samples, batch_size, out_dim)
    if out_dim > _P99_T1:
        p99_t1 = pred_real_mc[:, :, _P99_T1].mean(axis=0)
        p99_end = min(out_dim, _P99_SLICE.stop)
        if p99_end > _P99_T1:
            p99_horizon = pred_real_mc[:, :, _P99_T1:p99_end]
            viol_prob = (np.max(p99_horizon, axis=2) >= float(args.qos)).mean(axis=0)
        else:
            viol_prob = (pred_real_mc[:, :, _P99_T1] >= float(args.qos)).mean(axis=0)
    else:
        p99_t1 = pred_real_mc[:, :, 0].mean(axis=0)
        viol_prob = np.zeros(batch_size, dtype=np.float64)
    return np.column_stack([p99_t1, viol_prob])


def _predict(info):
    batch_size = len(info['next_info'])

    sys_sur, lat_sur, nxt_sur = _build_raw_features(info, sur_nxt_horizon)
    x_sur = _prepare_x(
        sys_sur, lat_sur, nxt_sur, sur_scaler_sys, sur_scaler_lat, sur_scaler_nxt, sur_top_indices
    )
    sur_pred, _, uncertain_mask = _surrogate_predict_with_cp(x_sur)

    result = np.array(sur_pred, dtype=np.float64)
    bnn_indices = np.arange(batch_size) if not cp_enabled else np.where(uncertain_mask)[0]

    if len(bnn_indices) > 0:
        sub_info = {'sys_data': info['sys_data'], 'next_info': [info['next_info'][i] for i in bnn_indices]}
        sys_bnn, lat_bnn, nxt_bnn = _build_raw_features(sub_info, bnn_nxt_horizon)
        x_bnn = _prepare_x(
            sys_bnn, lat_bnn, nxt_bnn, bnn_scaler_sys, bnn_scaler_lat, bnn_scaler_nxt, bnn_top_indices
        )
        bnn_pred = _bnn_predict_from_x(x_bnn)
        for j, idx in enumerate(bnn_indices):
            result[idx, 0] = bnn_pred[j, 0]
            result[idx, 1] = bnn_pred[j, 1]

    formatted = []
    for i in range(batch_size):
        formatted.append([round(float(result[i, 0]), 2), round(float(result[i, 1]), 3)])
    return formatted


def main():
    _load_surrogate()
    _load_bnn()
    _load_calibration()

    local_serv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_serv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    local_serv_sock.bind(('0.0.0.0', ServerPort))
    local_serv_sock.listen(1024)
    host_sock, _ = local_serv_sock.accept()
    host_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    logging.info('master connected')

    msg_buffer = ''
    terminate = False
    while True:
        data = host_sock.recv(2048).decode('utf-8')
        if len(data) == 0:
            logging.warning('connection reset by host, exiting...')
            break
        msg_buffer += data
        while '\n' in msg_buffer:
            cmd, msg_buffer = msg_buffer.split('\n', 1)
            if cmd.startswith('pred----'):
                info = json.loads(cmd.split('----')[-1])
                pred = _predict(info)
                host_sock.sendall(('pred----' + json.dumps(pred) + '\n').encode('utf-8'))
            elif cmd.startswith('terminate'):
                host_sock.sendall('experiment_done\n'.encode('utf-8'))
                terminate = True
                break
            else:
                logging.error('Unknown cmd format: %s', cmd)
                terminate = True
                break
        if terminate:
            break

    host_sock.close()
    local_serv_sock.close()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    main()
