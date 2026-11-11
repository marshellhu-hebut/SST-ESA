import argparse
from pathlib import Path
import numpy as np
import torch
from model import build_model, ARCHITECTURES

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(ARCHITECTURES), required=True)
    p.add_argument('--variant', choices=['compact_esa', 'grouped_summary'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--cycles', action='store_true')
    a = p.parse_args()
    x = np.asarray(np.load(a.input, allow_pickle=False), dtype=np.float32)
    c = ARCHITECTURES[a.dataset]
    if x.ndim != 4 or x.shape[1:] != (c['patch_count'], 14, c['input_patch']) or len(x) == 0 or (not np.isfinite(x).all()) or (a.batch_size < 1):
        raise ValueError('Invalid normalized input shape, finite values, or batch size')
    model = build_model(a.dataset, a.variant, a.checkpoint, a.device).eval()
    outputs = []
    with torch.inference_mode():
        for i in range(0, len(x), a.batch_size):
            outputs.append(model(torch.from_numpy(x[i:i + a.batch_size]).to(a.device)).cpu().numpy())
    y = np.concatenate(outputs) * (125 if a.cycles else 1)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.output, y)
    print(f'Saved {len(y)} predictions to {a.output}')
if __name__ == '__main__':
    main()
