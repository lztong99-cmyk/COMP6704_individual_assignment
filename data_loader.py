import numpy as np
import os
from sklearn.datasets import load_breast_cancer, fetch_rcv1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# ----------------------
# 1. PyTorch MNIST Loader (No Manual Downloads)
# ----------------------
def load_mnist_pytorch(binary_digits=(3, 8)):
    """Load MNIST via PyTorch (convert to NumPy for compatibility)."""
    from torchvision import datasets, transforms
    
    # Transform: Convert to tensor → flatten → normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] → [0, 1] float tensor (shape: [1,28,28])
        transforms.Lambda(lambda x: x.numpy().flatten())  # Flatten to 784D vector
    ])
    
    # Load train/test sets (cached locally after first download)
    train_dataset = datasets.MNIST(
        root='./datasets/pytorch_mnist',
        train=True,
        download=True,  # Downloads once to ./datasets/pytorch_mnist/
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./datasets/pytorch_mnist',
        train=False,
        download=True,
        transform=transform
    )
    
    # Convert PyTorch Dataset to NumPy arrays
    X_train = np.array([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    X_test = np.array([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
    
    # Combine and filter for binary classification (e.g., 3 vs 8)
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    digit1, digit2 = binary_digits
    mask = (y == digit1) | (y == digit2)
    X, y = X[mask], y[mask]
    y = np.where(y == digit1, -1, 1)  # Map to -1/1 for logistic regression
    
    return X, y

# ----------------------
# 2. Local RCV1 Loader (Pre-saved to Disk)
# ----------------------
def save_rcv1_local(save_dir="./datasets/rcv1/", n_samples=10000):
    """Download RCV1 once and save subset locally (run once)."""
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, "rcv1_X.npy")):
        return  # Already saved
    
    print("Downloading RCV1 (first run only)...")
    data = fetch_rcv1(subset='train')
    X = data.data[:n_samples].toarray()  # Use first 10k samples (dense array)
    y = data.target[:n_samples].toarray()[:, 0]  # Binary label (first class)
    np.save(os.path.join(save_dir, "rcv1_X.npy"), X)
    np.save(os.path.join(save_dir, "rcv1_y.npy"), y)
    print(f"RCV1 subset saved to {save_dir}")

# ----------------------
# Main Dataset Loader (Unified Interface)
# ----------------------
def load_rcv1_local(load_dir="./datasets/rcv1/"):
    """Load RCV1 from local .npy files (with safe preprocessing)."""
    save_rcv1_local(load_dir)  # Ensure data exists
    X = np.load(os.path.join(load_dir, "rcv1_X.npy"))
    y = np.load(os.path.join(load_dir, "rcv1_y.npy"))
    y = np.where(y == 0, -1, 1)  # Map to -1/1
    
    # Fix: Use a lower variance threshold (or skip for small subsets)
    var_threshold = 0.001  # Lower threshold (or set to 0 to disable)
    if np.var(X, axis=0).max() > var_threshold:
        X = VarianceThreshold(threshold=var_threshold).fit_transform(X)
    else:
        print("Warning: RCV1 subset has low variance—skipping feature filtering")
    
    return X, y

def load_dataset(name):
    """Load dataset by name: 'mnist_pytorch', 'rcv1_local', or 'breast_cancer'."""
    if name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        y = np.where(y == 0, -1, 1)
        scaler = StandardScaler()
        
    elif name == "mnist_pytorch":
        X, y = load_mnist_pytorch(binary_digits=(3, 8))
        scaler = None  # Already normalized to [0,1]
        
    elif name == "rcv1_local":
        X, y = load_rcv1_local()
        # Fix: Use StandardScaler with mean=True (dense data now)
        scaler = StandardScaler(with_mean=True)
        
    else:
        raise ValueError(f"Dataset {name} not supported! Use: 'mnist_pytorch', 'rcv1_local', 'breast_cancer'")
    
    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply scaling (skip for MNIST)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test