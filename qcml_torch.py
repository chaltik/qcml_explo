import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging


def generate_fuzzy_sphere(num_points=1000, noise_level=0.1):
    """Generates points on a 2D sphere embedded in 3D with optional noise."""
    phi = np.random.uniform(0, np.pi, num_points)  # Polar angle
    theta = np.random.uniform(0, 2 * np.pi, num_points)  # Azimuthal angle
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Introduce Gaussian noise
    noise = np.random.normal(0, noise_level, (num_points, 3))
    points = np.vstack([x, y, z]).T + noise

    return points

def generate_fuzzy_torus(num_points=1000, R=2.0, r=0.5, noise_level=0.1):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    noise = np.random.normal(0, noise_level, (num_points, 3))
    points = np.vstack([x, y, z]).T + noise

    return points

def initialize_matrix_configuration(dim=3, hilbert_dim=4):
    """Generates random Hermitian matrices as initial observables."""
    A = [np.random.randn(hilbert_dim, hilbert_dim) + 1j * np.random.randn(hilbert_dim, hilbert_dim) for _ in range(dim)]
    A = [0.5 * (M + M.conj().T) for M in A]  # Make matrices Hermitian
    return A

def plot_manifold(original_data, processed_data=None):
    """Plots original noisy sphere vs. quantum-processed data."""
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original data (red)
    ax.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], 
               c='red', alpha=0.5, label="Original Data", s=5)
    title = "Original Data"
    if processed_data is not None:
        # Plot processed quantum manifold (blue)
        ax.scatter(processed_data[:, 0], processed_data[:, 1], processed_data[:, 2], 
                c='blue', alpha=0.8, label="Quantum Processed", s=5)
        title="Original vs. Quantum-Processed Data"
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.show()

class HermitianMatrixLayer(nn.Module):
    def __init__(self, dim, num_matrices):
        super(HermitianMatrixLayer, self).__init__()
        self.dim = dim
        self.num_matrices = num_matrices
        
        self.diagonal = nn.Parameter(torch.randn(num_matrices, dim, dtype=torch.float32))
        self.upper_triangle_real = nn.Parameter(
            torch.randn(num_matrices, dim, dim, dtype=torch.float32).triu(1)
        )
        self.upper_triangle_imag = nn.Parameter(
            torch.randn(num_matrices, dim, dim, dtype=torch.float32).triu(1)
        )

    def forward(self):
        """
        Constructs Hermitian matrices from parameters.

        Returns:
            list[torch.Tensor]: List of Hermitian matrices.
        """
        hermitian_matrices = []
        for i in range(self.num_matrices):
            diagonal = torch.diag(self.diagonal[i])
            upper_triangle = self.upper_triangle_real[i] + 1j * self.upper_triangle_imag[i]
            full_matrix = diagonal + upper_triangle + upper_triangle.T.conj()
            hermitian_matrices.append(full_matrix)
        return hermitian_matrices

class EnergyLoss(nn.Module):
    def __init__(self, lambda_reg=0.1):
        super(EnergyLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.lambda_reg = lambda_reg

    def forward(self, A, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        N, D = X.shape
        dim = A[0].shape[0]

        A_tensor = torch.stack(A)  # (D, dim, dim)
        X_tensor = X.to(torch.complex64)

        I = torch.eye(dim, dtype=torch.complex64, device=X_tensor.device)
        I_exp = I.unsqueeze(0).unsqueeze(0).expand(N, D, dim, dim)

        X_cmplx = X_tensor.unsqueeze(-1).unsqueeze(-1)  # (N, D, 1, 1)
        scaled_I = X_cmplx * I_exp

        A_exp = A_tensor.unsqueeze(0).expand(N, -1, -1, -1)  # (N, D, dim, dim)
        H_diff = A_exp - scaled_I
        H_x = torch.einsum('ndij,ndkj->ndik', H_diff, H_diff.conj())
        H_x_final = 0.5 * H_x.sum(1)  # (N, dim, dim)

        eigvals, eigvecs = torch.linalg.eigh(H_x_final)  # (N, dim), (N, dim, dim)
        psi_0 = eigvecs[:, :, 0]  # (N, dim)

        psi_0_col = psi_0.unsqueeze(-1)  # (N, dim, 1)
        psi_0_adj = psi_0.unsqueeze(1).conj()  # (N, 1, dim)

        A_tensor_exp = A_tensor.unsqueeze(0).expand(N, -1, -1, -1)  # (N, D, dim, dim)

        A_psi = torch.matmul(A_tensor_exp, psi_0_col.unsqueeze(1))  # (N, D, dim, 1)
        A_psi = A_psi.squeeze(-1)  # (N, D, dim)

        position_vector = torch.matmul(psi_0_adj.unsqueeze(2), A_psi.unsqueeze(-1))
        position_vector = position_vector.squeeze(-1).squeeze(-1).real  # (N, D)

        position_error = ((position_vector - X_tensor.real) ** 2).sum()

        A_psi_norm_sq = (A_psi.conj() * A_psi).sum(-1).real  # (N, D)
        position_vector_sq = position_vector ** 2
        variance = (A_psi_norm_sq - position_vector_sq).sum()

        loss = position_error + self.lambda_reg * variance
        return loss / N

# Training wrapper
def train_model(dim, X, val_X=None, max_iter=400, learning_rate=0.01, patience=10, min_delta=1e-5, lambda_reg=0.1):
    num_matrices = X.shape[1]
    layer = HermitianMatrixLayer(dim, num_matrices)
    criterion = EnergyLoss(lambda_reg=lambda_reg)

    optimizer = torch.optim.Adam(layer.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-5, verbose=True
    )

    loss_trajectory = []
    val_loss_trajectory = []
    best_loss = float('inf')
    best_params = None
    patience_counter = 0

    for iteration in range(max_iter):
        optimizer.zero_grad()
        matrices = layer()
        loss = criterion(matrices, X)
        loss.backward()
        optimizer.step()

        scheduler.step(loss)

        loss_value = loss.item()
        loss_trajectory.append(loss_value)

        if val_X is not None:
            with torch.no_grad():
                val_loss = criterion(matrices, val_X).item()
            val_loss_trajectory.append(val_loss)

        if loss_value < best_loss - min_delta:
            best_loss = loss_value
            best_params = {name: param.clone().detach() for name, param in layer.named_parameters()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping at iteration {iteration}. Best loss: {best_loss:.4f}")
            break

    if best_params is not None:
        with torch.no_grad():
            for name, param in layer.named_parameters():
                param.copy_(best_params[name])

    return layer.forward(), loss_trajectory, val_loss_trajectory if val_X is not None else loss_trajectory



def torch_to_numpy(data):
    """
    Recursively converts PyTorch tensors (or nested lists of tensors) to NumPy arrays.
    
    Args:
        data: A PyTorch tensor, list of tensors, or nested structure containing tensors.
        
    Returns:
        A NumPy array or a nested structure of NumPy arrays.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()  # Detach from graph and move to CPU before conversion
    elif isinstance(data, (list, tuple)):
        return [torch_to_numpy(item) for item in data]  # Recursively handle lists or tuples
    elif isinstance(data, dict):
        return {key: torch_to_numpy(value) for key, value in data.items()}  # Handle dictionaries
    else:
        return data  # If not a tensor, return as is

def compute_quantum_metric(A, X):
    """
    Computes the quantum metric g(x) per the paper's definition.
    """
    quantum_metrics = []
    estimated_data = []
    estimated_eigvals = []
    printed=False
    for x in X:
        # Hamiltonian H(x)
        H_x = sum((A[i] - x[i] * np.eye(A[i].shape[0])) @ (A[i] - x[i] * np.eye(A[i].shape[0])) for i in range(len(A)))
        eigvals, eigvecs = np.linalg.eigh(H_x)
        psi_0 = eigvecs[:, 0]
        estimated_eigvals.append(eigvals)

        g_x = np.zeros((len(A), len(A)), dtype=np.float64)

        if not printed:
            logging.debug(f'x: {x}')
            logging.debug(f'psi0(x): {psi_0}')
            logging.debug(f'H_x shape: {H_x.shape}')
            logging.debug(f"N of A's: {len(A)}")
            logging.debug(f'D: {g_x.shape[0]}')
            printed=True
        for mu in range(len(A)):
            for nu in range(len(A)):
                total = 0.0
                for n in range(1, len(eigvals)):  # sum over excited states only
                    e_n = eigvals[n]
                    e_0 = eigvals[0]

                    if np.abs(e_n - e_0) < 1e-12:
                        continue  # avoid division by zero or degenerate case

                    psi_n = eigvecs[:, n]
                    term1 = np.vdot(psi_0, A[mu] @ psi_n)
                    term2 = np.vdot(psi_n, A[nu] @ psi_0)
                    total += (term1 * term2) / (e_n - e_0)

                g_x[mu, nu] = 2 * np.real(total)

        quantum_metrics.append(g_x)

        # Estimated data point
        x_est = [np.real(psi_0.conj().T @ A[i] @ psi_0) for i in range(len(A))]
        estimated_data.append(x_est)

    return quantum_metrics, np.array(estimated_data), np.array(estimated_eigvals)



def compute_intrinsic_dimension(quantum_metrics):
    estimated_dims = []
    metric_eigvals = []
    for g_x in quantum_metrics:
        D = g_x.shape[0]
        eigvals = np.linalg.eigvalsh(g_x)
        eigvals = np.sort(eigvals)
        metric_eigvals.append(eigvals)
        spectral_gaps = eigvals[1:] / (eigvals[:-1]+1e-10)
        gamma = np.argmax(spectral_gaps) + 1  # index of largest gap

        d_x = D - gamma
        estimated_dims.append(d_x)

    return estimated_dims, metric_eigvals


def main():
    logging.basicConfig(level=logging.INFO)
    hilbert_dim=4
    num_points = 5000
    #data = generate_fuzzy_torus(num_points, R=8, r=1, noise_level=0.05)
    data = generate_fuzzy_sphere(num_points,noise_level=0.15)
    train_data,val_data = train_test_split(data,test_size=0.2)

    optimized_matrices, train_loss, val_loss = train_model(hilbert_dim,train_data, val_X=val_data,lambda_reg=0.001)
    for A in (torch_to_numpy(optimized_matrices)):
        print(A)

    plt.figure()
    plt.plot(train_loss,color='b',label='Train')
    plt.plot(val_loss,color='g',label='Val')
    plt.legend()
    plt.grid()
    plt.show()

    quantum_metrics, estimated_data, estimated_eigvals = compute_quantum_metric(torch_to_numpy(optimized_matrices), data)

    plot_manifold(train_data,estimated_data)

    # Estimate intrinsic dimensions
    intrinsic_dimensions, metrics_eigvals = compute_intrinsic_dimension(quantum_metrics)

    plt.hist(intrinsic_dimensions,bins=30)
    plt.title('Intrinsic dim distribution')
    plt.show()
    # Compute the final estimated intrinsic dimension as the mode of all estimates
    estimated_dimension = max(set(intrinsic_dimensions), key=intrinsic_dimensions.count)

    # Display estimated intrinsic dimension
    logging.info(estimated_dimension)

if __name__=="__main__":
    main()