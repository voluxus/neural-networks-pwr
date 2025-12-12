from sklearn.model_selection import train_test_split
from network.train import train_and_evaluate
from preprocessing.preprocessing import PreprocessingPipeline


def experiment_hidden_layers(X_train, y_train, X_test, y_test):
    """Test different hidden layer configurations."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Different Hidden Layer Dimensions")
    print("="*80)
    
    results = []
    for hidden_config in [list()] + [[2 ** i] for i in range(1, 11)]:
        result = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            hidden_layers=hidden_config,
            activation='relu',
            learning_rate=0.1,
            epochs=100,
            normalize=True
        )
        results.append(result)
    return results


def experiment_learning_rates(X_train, y_train, X_test, y_test):
    """Test different learning rates."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Different Learning Rates")
    print("="*80)
    
    results = []
    for lr in [0.001, 0.01, 0.1, 0.5]:
        result = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            hidden_layers=[64, 32],
            activation='relu',
            learning_rate=lr,
            epochs=50,
            normalize=True
        )
        results.append(result)
    return results


def experiment_weight_init(X_train, y_train, X_test, y_test):
    """Test different weight initialization standard deviations."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Different Weight Initialization Std")
    print("="*80)
    
    results = []
    for std in [0.01, 0.1, 0.5, 1.0]:
        result = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            hidden_layers=[64, 32],
            activation='relu',
            learning_rate=0.01,
            epochs=50,
            normalize=True,
            weight_init_std=std
        )
        results.append(result)
    return results


def experiment_normalization(X_train, y_train, X_test, y_test):
    """Test normalized vs unnormalized data."""
    print("\n" + "="*80)
    print("EXPERIMENT 4: Normalized vs Unnormalized Data")
    print("="*80)
    
    results = []
    for normalize in [False, True]:
        result = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            hidden_layers=[64, 32],
            activation='relu',
            learning_rate=0.01,
            epochs=50,
            normalize=normalize
        )
        results.append(result)
    return results


def experiment_layer_depth(X_train, y_train, X_test, y_test):
    """Test different number of layers."""
    print("\n" + "="*80)
    print("EXPERIMENT 5: Different Number of Layers")
    print("="*80)
    
    results = []
    for layers in [[64], [64, 32], [64, 32, 16], [128, 64, 32, 16]]:
        result = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            hidden_layers=layers,
            activation='relu',
            learning_rate=0.01,
            epochs=50,
            normalize=True
        )
        results.append(result)
    return results


def print_summary(results):
    """Print summary of all experiments."""
    print("\n" + "="*80)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*80)
    for i, res in enumerate(results, 1):
        print(f"{i}. Hidden: {res['hidden_layers']}, LR: {res['learning_rate']}, "
              f"Norm: {res['normalize']}, Std: {res['weight_init_std']}, "
              f"Train Acc: {res['train_accuracy']:.4f}, Test Acc: {res['test_accuracy']:.4f}")


if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    pipeline = PreprocessingPipeline()
    features, targets = pipeline.run()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42, stratify=targets
    )
    
    print(f"Dataset shape: {features.shape}")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Toggle experiments on/off here
    all_results = []
    
    all_results.extend(experiment_hidden_layers(X_train, y_train, X_test, y_test))
    #all_results.extend(experiment_learning_rates(X_train, y_train, X_test, y_test))
    #all_results.extend(experiment_weight_init(X_train, y_train, X_test, y_test))
    #all_results.extend(experiment_normalization(X_train, y_train, X_test, y_test))
    #all_results.extend(experiment_layer_depth(X_train, y_train, X_test, y_test))
    
    # Summary
    print_summary(all_results)