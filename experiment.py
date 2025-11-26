import numpy as np
import matplotlib.pyplot as plt
from .models import LogisticRegression
from .optimizers import UnlearningOptimizers

class ExperimentRunner:
    def __init__(self, dataset_name, lambda_reg=1e-3):
        self.dataset_name = dataset_name
        self.lambda_reg = lambda_reg
        self.convergence_data = {}
    
    def run(self, X_train, X_test, y_train, y_test):
        """Run with debugged methods (no threshold line)."""
        baseline = LogisticRegression(lambda_reg=self.lambda_reg)
        baseline.fit(X_train, y_train)
        print(f"Baseline accuracy: {baseline.accuracy(X_test, y_test):.4f}")
        
        optimizers = UnlearningOptimizers(baseline, X_train, y_train)
        idx = np.random.randint(0, len(X_train))
        
        # Run debugged methods
        _, newton_conv = optimizers.newton_unlearning_debug(idx)
        _, lbfgs_conv = optimizers.lbfgs_unlearning_debug(idx)
        _, gd_conv = optimizers.gd_unlearning_debug(idx)
        
        self.convergence_data = {
            "newton": newton_conv,
            "lbfgs": lbfgs_conv,
            "gd": gd_conv
        }
        
        # Verify iteration counts
        print(f"Newton iterations: {len(newton_conv['grad_norms'])}")
        print(f"L-BFGS iterations: {len(lbfgs_conv['grad_norms'])}")
        print(f"GD iterations: {len(gd_conv['grad_norms'])}")
        
        self._plot_clean_convergence()
    
    def _plot_clean_convergence(self):
        """Clean plots without threshold line."""
        max_iter = 20
        iterations_full = list(range(1, max_iter + 1))
        
        method_configs = {
            "newton": {"label": "Newton", "color": "#1f77b4", "marker": "o"},
            "lbfgs": {"label": "L-BFGS", "color": "#2ca02c", "marker": "s"},
            "gd": {"label": "Plain GD", "color": "#ff7f0e", "marker": "^"}
        }
        
        # ----------------------
        # Gradient Norm Plot (No Threshold)
        # ----------------------
        plt.figure(figsize=(10, 6))
        for method, config in method_configs.items():
            data = self.convergence_data[method]
            grad_norms = data["grad_norms"]
            
            # Pad to 20 iterations
            if len(grad_norms) < max_iter:
                grad_norms += [grad_norms[-1]] * (max_iter - len(grad_norms))
            
            plt.plot(
                iterations_full,
                grad_norms,
                label=config["label"],
                color=config["color"],
                marker=config["marker"],
                markersize=7,
                linewidth=2.5
            )
        
        # Formatting (no threshold line)
        plt.xlim(1, max_iter)
        plt.xticks(range(1, max_iter + 1, 2))
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel("Gradient Norm (Log Scale)", fontsize=12)
        plt.yscale("log")
        plt.title(f"Convergence Curves: Newton vs. L-BFGS vs. Plain GD ({self.dataset_name})", fontsize=14)
        plt.legend(fontsize=11, loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/{self.dataset_name}_clean_convergence.png", dpi=300)
        plt.close()
        
        # ----------------------
        # Parameter Distance Plot (No Threshold)
        # ----------------------
        plt.figure(figsize=(10, 6))
        for method, config in method_configs.items():
            data = self.convergence_data[method]
            param_dist = data["param_distances"]
            
            if len(param_dist) < max_iter:
                param_dist += [param_dist[-1]] * (max_iter - len(param_dist))
            
            plt.plot(
                iterations_full,
                param_dist,
                label=config["label"],
                color=config["color"],
                marker=config["marker"],
                markersize=7,
                linewidth=2.5
            )
        
        plt.xlim(1, max_iter)
        plt.xticks(range(1, max_iter + 1, 2))
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel("Parameter Distance from Original Model ($L_2$-Norm)", fontsize=12)
        plt.title(f"Parameter Update: Newton vs. L-BFGS vs. Plain GD ({self.dataset_name})", fontsize=14)
        plt.legend(fontsize=11, loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/{self.dataset_name}_clean_params.png", dpi=300)
        plt.close()