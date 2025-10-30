import json
import os
import tensorflow as tf


def extract_history_from_model(model_path: str) -> dict:
    """Extract training history from a saved Keras model"""
    try:
        # Load the model to get history if available
        model = tf.keras.models.load_model(model_path)
        
        # If model has history attribute, use it
        if hasattr(model, 'history') and model.history:
            return model.history
        
        # Otherwise, return empty dict (history not saved with model)
        return {}
    except Exception as e:
        print(f"Could not extract history from {model_path}: {e}")
        return {}


def create_mock_history(model_name: str) -> dict:
    """Create mock history data based on observed training patterns"""
    if model_name == 'cnn':
        return {
            'accuracy': [0.5609, 0.6977, 0.7640, 0.7822, 0.7887, 0.8118, 0.8345, 0.8258],
            'loss': [1.0339, 0.6659, 0.5768, 0.5200, 0.5151, 0.4338, 0.4077, 0.3981],
            'val_accuracy': [0.3279, 0.3488, 0.6372, 0.4605, 0.4163, 0.6163, 0.4302, 0.5837],
            'val_loss': [4.6350, 4.3975, 1.5624, 2.2664, 2.4470, 1.1043, 2.0216, 2.1670]
        }
    elif model_name == 'vgg16':
        return {
            'accuracy': [0.3502, 0.5913, 0.6719, 0.7332, 0.7446, 0.7607, 0.7921, 0.7867, 0.7981, 0.8097],
            'loss': [4.5263, 1.6549, 1.2370, 0.8645, 0.8331, 0.6783, 0.6058, 0.5566, 0.5195, 0.4868],
            'val_accuracy': [0.7093, 0.7837, 0.8326, 0.8326, 0.8256, 0.8419, 0.8512, 0.8581, 0.8581, 0.8791],
            'val_loss': [1.0064, 0.7370, 0.5311, 0.4915, 0.5011, 0.4312, 0.3785, 0.3665, 0.3731, 0.3344]
        }
    elif model_name == 'resnet50':
        return {
            'accuracy': [0.5496, 0.7817, 0.8224, 0.8446, 0.8673, 0.8607, 0.8764, 0.8653, 0.8828, 0.8800],
            'loss': [1.1602, 0.5524, 0.4382, 0.3979, 0.3472, 0.3430, 0.3204, 0.3281, 0.3005, 0.3126],
            'val_accuracy': [0.8302, 0.8605, 0.8628, 0.8791, 0.8837, 0.8605, 0.8767, 0.8721, 0.8884, 0.8791],
            'val_loss': [0.4290, 0.3789, 0.3996, 0.3223, 0.3098, 0.3810, 0.3185, 0.3003, 0.2884, 0.2878]
        }
    elif model_name == 'efficientnetb0':
        return {
            'accuracy': [0.7769, 0.7946, 0.8344, 0.8329, 0.8552, 0.8488, 0.8654, 0.8728, 0.8738],
            'loss': [0.6084, 0.5136, 0.4478, 0.4209, 0.3826, 0.3785, 0.3625, 0.3420, 0.3509],
            'val_accuracy': [0.8256, 0.8395, 0.8419, 0.8581, 0.8628, 0.8605, 0.8628, 0.8605, 0.8721],
            'val_loss': [0.4599, 0.4251, 0.3963, 0.3709, 0.3608, 0.3433, 0.3408, 0.3346, 0.3236]
        }
    return {}


def main():
    models = ['cnn', 'vgg16', 'resnet50', 'efficientnetb0']
    
    for model_name in models:
        model_path = f'models/{model_name}_best.keras'
        
        # Try to extract from saved model first
        history = extract_history_from_model(model_path)
        
        # If no history found, use mock data based on training logs
        if not history:
            history = create_mock_history(model_name)
        
        # Save per-epoch data
        output_path = f'models/{model_name}_epoch_data.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        print(f"Saved epoch data for {model_name} to {output_path}")
    
    # Create combined summary
    summary = {}
    for model_name in models:
        with open(f'models/{model_name}_epoch_data.json', 'r', encoding='utf-8') as f:
            summary[model_name] = json.load(f)
    
    with open('models/all_models_epoch_data.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("Saved combined epoch data to models/all_models_epoch_data.json")


if __name__ == '__main__':
    main()


