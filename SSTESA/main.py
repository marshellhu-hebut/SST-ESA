import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import torch
from model import ARCHITECTURES, FIXED_SEEDS
from training import Trainer, Train, make_args
DATASETS = ('FD001', 'FD002', 'FD003', 'FD004')
DATA_DIR = None
OUTPUT_DIR = Path(__file__).resolve().parent / 'results'

def find_data_dir(value, datasets):
    candidates = []
    if value is not None:
        candidates.append(Path(value).expanduser())
    elif os.environ.get('CMAPSS_DATA_DIR'):
        candidates.append(Path(os.environ['CMAPSS_DATA_DIR']).expanduser())
    else:
        here = Path(__file__).resolve().parent
        for base in dict.fromkeys([here, Path.cwd(), *list(here.parents)[:4]]):
            candidates.extend([base / 'CMAPSSData', base / 'CMAPSS', base / 'data' / 'CMAPSSData', base / 'data' / 'CMAPSS', base])
    for path in candidates:
        if all(((path / f'{name}_{ds}.txt').is_file() for ds in datasets for name in ('train', 'test', 'RUL'))):
            return path.resolve()
    raise FileNotFoundError('C-MAPSS files not found. Set DATA_DIR or use --data-dir.')

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=list(ARCHITECTURES), default=list(DATASETS))
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    data_dir = find_data_dir(args.data_dir, args.datasets)
    if args.check_data:
        for dataset in args.datasets:
            trainer = Train(make_args(dataset), data_dir, args.device)
            print(json.dumps(dict(dataset=dataset, seed=FIXED_SEEDS[dataset], train_shape=list(trainer.train_data.shape), validation_shape=list(trainer.val_data.shape), test_shape=list(trainer.test_data.shape), parameters=sum((p.numel() for p in trainer.net.parameters())))), flush=True)
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return
    session = args.output_dir.expanduser().resolve() / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    session.mkdir(parents=True, exist_ok=False)
    results = []
    for dataset in args.datasets:
        trainer = Trainer(dataset, data_dir, session / dataset, args.device)
        results.append(trainer.run())
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (session / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(str(session), flush=True)
if __name__ == '__main__':
    main()
