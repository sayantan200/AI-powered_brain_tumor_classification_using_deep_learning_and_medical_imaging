import argparse
import json
import os
import subprocess


def run(cmd: list[str]):
    print('> ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224])
    args = parser.parse_args()

    models = ['cnn', 'vgg16', 'resnet50', 'efficientnetb0']
    summary = {}
    for m in models:
        cmd = ['python', 'main_train.py', '--model', m,
               '--epochs', str(args.epochs),
               '--batch_size', str(args.batch_size),
               '--img_size', str(args.img_size[0]), str(args.img_size[1])]
        run(cmd)

        metrics_file = os.path.join('models', f'{m}_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                summary[m] = json.load(f)

    with open(os.path.join('models', 'summary_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('Saved metrics summary to models/summary_metrics.json')


if __name__ == '__main__':
    main()




