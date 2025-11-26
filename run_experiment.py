from src.data_loader import load_dataset
from src.experiment import ExperimentRunner

if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    
    # Choose dataset (e.g., 'mnist_pytorch' or 'breast_cancer')
    dataset_name = 'rcv1_local'
    X_train, X_test, y_train, y_test = load_dataset(dataset_name)
    
    # Run simplified experiment
    experiment = ExperimentRunner(dataset_name=dataset_name, lambda_reg=1e-3)
    experiment.run(X_train, X_test, y_train, y_test)
    
    print(f"Simplified convergence plots saved to results/{dataset_name}_simplified_*.png")