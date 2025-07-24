import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging

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

def calc_position_and_variance(A, X):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)

    N, D = X.shape
    dim = A[0].shape[0]
    if isinstance(A[0], np.ndarray): 
        A_tensor = torch.stack([torch.tensor(a, dtype=torch.complex64) for a in A])
    else:
        A_tensor = torch.stack(A)  # (D, dim, dim)
    X_tensor = X.to(torch.complex64)

    I = torch.eye(dim, dtype=torch.complex64, device=X_tensor.device)
    I_exp = I.unsqueeze(0).unsqueeze(0).expand(N, D, dim, dim)

    X_cmplx = X_tensor.unsqueeze(-1).unsqueeze(-1)  # (N, D, 1, 1)
    scaled_I = X_cmplx * I_exp
    valid = torch.isfinite(X_tensor)
    valid_exp = valid.unsqueeze(-1).unsqueeze(-1)
    A_exp = A_tensor.unsqueeze(0).expand(N, -1, -1, -1)  # (N, D, dim, dim)
    scaled_I_clean = torch.where(valid_exp, scaled_I, A_exp)
    H_diff = A_exp - scaled_I_clean
    H_diff = H_diff * valid_exp
    H_x = torch.einsum('ndij,ndkj->ndik', H_diff, H_diff.conj())
    H_x_final = 0.5 * H_x.sum(1)  # (N, dim, dim)
    eps = 1e-5
    H_x_final += eps * torch.eye(dim, dtype=torch.complex64, device=X_tensor.device)

    eigvals, eigvecs = torch.linalg.eigh(H_x_final)
    psi_0 = eigvecs[:, :, 0]
    phase = psi_0[:, 0] / psi_0[:, 0].abs()
    psi_0 = psi_0 / phase.unsqueeze(-1)

    projector = torch.einsum('ni,nj->nij', psi_0, psi_0.conj())

    A_tensor_exp = A_tensor.unsqueeze(0).expand(N, -1, -1, -1)
    position_vector = torch.einsum('ndij,njk->nd', A_tensor_exp, projector).real

    A2_tensor_exp = torch.matmul(A_tensor_exp, A_tensor_exp)
    expectation_A2 = torch.einsum('ndij,njk->nd', A2_tensor_exp, projector).real
    variance = (expectation_A2 - position_vector ** 2).sum()

    return position_vector, variance

class EnergyLoss(nn.Module):
    def __init__(self, lambda_reg=0.1):
        super(EnergyLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, A, X):
        position_vector, variance = calc_position_and_variance(A, X)
        X_tensor = torch.tensor(X,dtype=torch.complex64)
        N = X.shape[0]
        position_error = ((position_vector - X_tensor.real) ** 2).sum()
        loss = position_error + self.lambda_reg * variance
        return loss / N
    


def check_hermitian(matrices, tol=1e-8):
    """
    Check if the given matrices are Hermitian.
    
    Args:
        matrices: List of NumPy arrays representing matrices.
        tol: Tolerance for numerical errors.
        
    Returns:
        hermitian_results: List of booleans indicating if each matrix is Hermitian.
    """
    hermitian_results = []
    for i, A in enumerate(matrices):
        is_hermitian = np.allclose(A, A.conj().T, atol=tol)
        hermitian_results.append(is_hermitian)
        if not is_hermitian:
            print(f"Matrix A[{i}] is not Hermitian!")
    return hermitian_results

def check_hermitian_batch(H_batch, atol=1e-5):
    """
    Checks if each matrix in a batch is Hermitian.
    
    Args:
        H_batch (torch.Tensor): shape (N, dim, dim), complex
        atol (float): tolerance

    Returns:
        torch.BoolTensor: shape (N,), indicating if each matrix is Hermitian
    """
    hermitian_diff = H_batch - H_batch.conj().transpose(-1, -2)
    is_hermitian = hermitian_diff.abs().max(dim=-1).values.max(dim=-1).values < atol
    return is_hermitian


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

        if iteration%50==0:
            print(f'Epoch {iteration}: loss = {loss_value}, val_loss = {val_loss}')
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

