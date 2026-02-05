# Generate GPU/predictor config for BNN surrogate (decision tree) predictor.
# Same structure as make_gpu_config.py but script = social_media_predictor_bnn_surrogate.py.
# Usage: python make_gpu_config_bnn_surrogate.py --gpu-config predictor_bnn_surrogate.json

import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu-config', dest='gpu_config', type=str, default='predictor_bnn_surrogate.json',
                    help='Output config filename in docker_swarm/config/')
args = parser.parse_args()

gpu_config_path = Path('..') / 'config' / args.gpu_config.strip()

gpu_config = {}
gpu_config['gpus'] = []  # No GPU needed for surrogate tree
gpu_config['ip_addr'] = 'gpu-server-ip'
gpu_config['host'] = 'tsg-gpu1.ece.cornell.edu'
gpu_config['working_dir'] = '/home/yz2297/sinan-local/surrogate'
gpu_config['script'] = 'social_media_predictor_bnn_surrogate.py'

with open(str(gpu_config_path), 'w+') as f:
    json.dump(gpu_config, f, indent=4, sort_keys=True)

print(f"Wrote {gpu_config_path} (script={gpu_config['script']})")
