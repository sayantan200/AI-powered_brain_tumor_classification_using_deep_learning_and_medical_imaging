import argparse
import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from utils.dataset_loader import make_datasets
from utils.evaluation import classification_report_from_probs
from utils.plotting import plot_training_curves, plot_confusion_matrix
from models.cnn_model import build_cnn
from models.vgg16_model import build_vgg16
from models.resnet_model import build_resnet50
from models.efficientnet_model import build_efficientnet_b0


def get_model(name: str, input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    name = name.lower()
    if name == 'cnn':
        return build_cnn(input_shape, num_classes)
    if name == 'vgg16':
        return build_vgg16(input_shape, num_classes)
    if name == 'resnet50':
        return build_resnet50(input_shape, num_classes)
    if name == 'efficientnetb0':
        return build_efficientnet_b0(input_shape, num_classes)
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(description='Train/Evaluate brain tumor classification models')
    parser.add_argument('--train_dir', type=str, default='dataset/Training')
    parser.add_argument('--test_dir', type=str, default='dataset/Testing')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'vgg16', 'resnet50', 'efficientnetb0'])
    parser.add_argument('--out_dir', type=str, default='models')
    args = parser.parse_args()

    data = make_datasets(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        image_size=tuple(args.img_size),
        batch_size=args.batch_size,
        val_split=args.val_split,
    )

    class_names = data['class_names']
    num_classes = len(class_names)
    input_shape = (args.img_size[0], args.img_size[1], 3)

    model = get_model(args.model, input_shape, num_classes)
    # Enhanced optimizer and learning rate scheduling
    if args.model == 'cnn':
        # Use lower learning rate for CNN from scratch
        initial_lr = 1e-4
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
        )
    else:
        # Higher learning rate for transfer learning models
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1
        )
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    # Get class distribution
    class_counts = []
    for class_name in class_names:
        class_dir = os.path.join(args.train_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts.append(count)
        else:
            class_counts.append(1)  # fallback
    
    class_weights = compute_class_weight('balanced', classes=np.unique(class_names), y=np.repeat(class_names, class_counts))
    class_weight_dict = {i: class_weights[i] for i in range(len(class_names))}
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f'{args.model}_best.keras')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        lr_scheduler,
        tf.keras.callbacks.CSVLogger(f'models/{args.model}_training_log.csv'),
    ]
    
    # Add TensorBoard for monitoring
    if args.model == 'cnn':
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=f'models/tensorboard/{args.model}',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ))

    history = model.fit(
        data['train'],
        validation_data=data['val'],
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict if args.model == 'cnn' else None,
    )
    # Save training curves
    plots_dir = os.path.join(args.out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_training_curves(history.history, os.path.join(plots_dir, f'{args.model}_curves.png'))

    # Evaluate on test set
    y_true = []
    y_prob = []
    for x_batch, y_batch in data['test']:
        probs = model.predict(x_batch, verbose=0)
        y_true.append(y_batch.numpy())
        y_prob.append(probs)
    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)

    report = classification_report_from_probs(y_true, y_prob, class_names)
    print('Test metrics:')
    for k, v in report.items():
        if k == 'confusion_matrix':
            print(f'{k}:')
            print(np.array(v))
        else:
            print(f'{k}: {v}')

    # Save confusion matrix image and metrics json
    cm_path = os.path.join(plots_dir, f'{args.model}_confusion.png')
    plot_training_curves(history.history, os.path.join(plots_dir, f'{args.model}_curves.png'))
    plot_confusion_matrix(np.array(report['confusion_matrix']), class_names, cm_path, normalize=True)
    import json
    with open(os.path.join(args.out_dir, f'{args.model}_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in report.items() if k != 'confusion_matrix'}, f, indent=2)


if __name__ == '__main__':
    main()


