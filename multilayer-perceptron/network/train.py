import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from network.network import Network


def train_and_evaluate(
    X_train, y_train, X_test, y_test,
    hidden_layers=[64, 32],
    activation='sigmoid',
    learning_rate=0.1,
    epochs=100,
    batch_size=32,
    weight_init_std=None,
    normalize=True,
    multiclass=False,
    seed=42
):
    """
    Train and evaluate a network configuration
    
    Returns:
        dict with training results
    """

    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Determine output size
    output_size = 1 if not multiclass else len(np.unique(y_train))
    
    # Create network
    network = Network(
        hidden_layers=hidden_layers,
        input_size=X_train.shape[1],
        output_size=output_size,
        activation=activation,
        multiclass=multiclass,
        learning_rate=learning_rate,
        weight_init_std=weight_init_std,
        seed=seed
    )
    
    # Train
    print(f"\nTraining network: hidden_layers={hidden_layers}, activation={activation}, "
          f"lr={learning_rate}, normalize={normalize}, weight_std={weight_init_std}")
    print("=" * 80)
    
    network.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=False)
    
    # Evaluate
    y_pred_train = network.predict_classes(X_train)
    y_pred_test = network.predict_classes(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_pred_test))
    
    return {
        'hidden_layers': hidden_layers,
        'activation': activation,
        'learning_rate': learning_rate,
        'normalize': normalize,
        'weight_init_std': weight_init_std,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'network': network
    }
