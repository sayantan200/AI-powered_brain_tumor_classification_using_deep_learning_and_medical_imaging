import os
import sys
from pprint import pprint

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import load_config, index_dataset, summarize_index, train_val_split


def main() -> int:
    cfg = load_config('config/data.yaml')

    print('Config:')
    print(f"  train_dir: {cfg.train_dir}")
    print(f"  test_dir:  {cfg.test_dir}")
    print(f"  classes:   {cfg.class_names or '[auto-discovery]'}")

    # Auto-discover classes if not specified
    if not cfg.class_names:
        # Refresh config with discovered classes
        from utils.data_loader import _discover_classes  # type: ignore
        classes = _discover_classes(cfg.train_dir)
        print(f"Discovered classes: {classes}")
        if not classes:
            print('No class folders found under train_dir.')
            return 1
        cfg.class_names = classes

    train_index = index_dataset(cfg.train_dir, cfg.class_names, cfg.allowed_exts)
    test_index = index_dataset(cfg.test_dir, cfg.class_names, cfg.allowed_exts)

    train_counts = summarize_index('train', train_index)
    test_counts = summarize_index('test', test_index)

    print('\nCounts per class:')
    print('Train:')
    pprint(train_counts)
    print('Test:')
    pprint(test_counts)

    # If no explicit val split, compute virtual split for reporting
    train_split, val_split = train_val_split(train_index, cfg.val_split, shuffle=False)
    print('\nProposed split (no files moved):')
    print('Train after split:')
    pprint({k: len(v) for k, v in train_split.items()})
    print('Val:')
    pprint({k: len(v) for k, v in val_split.items()})

    problems = []
    if train_counts.get('__total__', 0) == 0:
        problems.append('No training files found.')
    if test_counts.get('__total__', 0) == 0:
        problems.append('No test files found.')
    missing_train_classes = [c for c in cfg.class_names if not train_index.get(c)]
    missing_test_classes = [c for c in cfg.class_names if not test_index.get(c)]
    if missing_train_classes:
        problems.append(f"Missing train classes: {missing_train_classes}")
    if missing_test_classes:
        problems.append(f"Missing test classes: {missing_test_classes}")

    if problems:
        print('\nIssues detected:')
        for p in problems:
            print(f"- {p}")
        return 2

    print('\nDataset looks good.')
    return 0


if __name__ == '__main__':
    sys.exit(main())


