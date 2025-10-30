import argparse
import json
import os
import sys
import numpy as np
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset_loader import make_datasets
from utils.evaluation import classification_report_from_probs
from ensemble.ensemble_model import average_probabilities


def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', required=True, help='Paths to saved .keras models')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_dir', type=str, default='dataset/Training')
    parser.add_argument('--test_dir', type=str, default='dataset/Testing')
    args = parser.parse_args()

    data = make_datasets(args.train_dir, args.test_dir, image_size=tuple(args.img_size), batch_size=args.batch_size)
    class_names = data['class_names']

    loaded = [load_model(p) for p in args.models]

    y_true = []
    all_probs = [[] for _ in loaded]
    for x_batch, y_batch in data['test']:
        y_true.append(y_batch.numpy())
        for i, model in enumerate(loaded):
            probs = model.predict(x_batch, verbose=0)
            all_probs[i].append(probs)

    y_true = np.concatenate(y_true, axis=0)
    all_probs = [np.concatenate(p, axis=0) for p in all_probs]
    y_prob_ens = average_probabilities(all_probs)

    report = classification_report_from_probs(y_true, y_prob_ens, class_names)
    print('Ensemble metrics:')
    print(json.dumps({k: v for k, v in report.items() if k != 'confusion_matrix'}, indent=2))


if __name__ == '__main__':
    main()


