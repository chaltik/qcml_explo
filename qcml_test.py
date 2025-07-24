import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from qcml_tf import tf_to_numpy, train_model as train_tf, calc_position_and_variance as predict_tf
from qcml_torch import torch_to_numpy, train_model as train_torch, calc_position_and_variance as predict_torch
import logging
import click

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


def compute_quantum_metric(A, X, f='tf'):
    """
    Computes the quantum metric g(x) per the paper's definition.
    """
    quantum_metrics = []
    #estimated_data = []
    estimated_eigvals = []
    printed=False
    for x in X:
        # Hamiltonian H(x)
        H_x = sum((A[i] - x[i] * np.eye(A[i].shape[0])) @ (A[i] - x[i] * np.eye(A[i].shape[0])) for i in range(len(A)))
        eigvals, eigvecs = np.linalg.eigh(H_x)
        psi_0 = eigvecs[:, 0]
        estimated_eigvals.append(eigvals)

        g_x = np.zeros((len(A), len(A)), dtype=np.float64)

        # if not printed:
        #     logging.debug(f'A[0]: {A[0]}')
        #     logging.debug(f'x: {x}')
        #     logging.debug(f'psi0(x): {psi_0}')
        #     logging.debug(f'H_x shape: {H_x.shape}')
        #     logging.debug(f"N of A's: {len(A)}")
        #     logging.debug(f'D: {g_x.shape[0]}')
        #     printed=True
        for mu in range(len(A)):
            for nu in range(len(A)):
                total = 0.0
                for n in range(1, len(eigvals)):  # sum over excited states only
                    e_n = eigvals[n]
                    e_0 = eigvals[0]

                    if np.abs(e_n - e_0) < 1e-12:
                        continue  # avoid division by zero or degenerate case

                    psi_n = eigvecs[:, n]
                    if not printed:
                        print("psi_0 shape:", psi_0.shape)
                        print("A[mu] shape:", A[mu].shape)
                        print("psi_n shape:", psi_n.shape)
                        print("A[mu] @ psi_n shape:", (A[mu] @ psi_n).shape)
                        printed=True
                    term1 = np.vdot(psi_0, A[mu] @ psi_n)
                    term2 = np.vdot(psi_n, A[nu] @ psi_0)
                    total += (term1 * term2) / (e_n - e_0)

                g_x[mu, nu] = 2 * np.real(total)

        quantum_metrics.append(g_x)

        # Estimated data point
        x_est = [np.real(psi_0.conj().T @ A[i] @ psi_0) for i in range(len(A))]
        #estimated_data.append(x_est)
    if f=='tf':
        estimated_data, estimated_variance = predict_tf(A,X)
        estimated_data = estimated_data.numpy().real
    else:
        estimated_data, estimated_variance = predict_torch(A,X)
        estimated_data = torch_to_numpy(estimated_data)

    return quantum_metrics, estimated_data, np.array(estimated_eigvals)



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


@click.command()
@click.option('-m',default='sphere',help='"sphere"(default) or "torus"')
@click.option('-f',default='tf',help='"tf"(tensorflow)(default) or "torch"')
@click.option('--noise_level',default=0.1,help='noise level to add to sphere (radius=1) or torus (R=8,r=1), defau;t=0.1')
@click.option('--lambda_reg',default=0.001,help='L2 regularization (default=0.001)')
def main(m,f,noise_level,lambda_reg):
    logging.basicConfig(level=logging.INFO)
    hilbert_dim=4
    num_points = 5000
    if m=='sphere':
        data = generate_fuzzy_sphere(num_points,noise_level=noise_level)
    elif m=='torus':
        data = generate_fuzzy_torus(num_points, R=8, r=1, noise_level=noise_level)
    else:
        raise ValueError(f'invalid manifold "{m}"')

    train_data,val_data = train_test_split(data,test_size=0.2)

    tonumpy = lambda f: tf_to_numpy if f=='tf' else torch_to_numpy

    if f=='tf':
        optimized_matrices, train_loss, val_loss = train_tf(hilbert_dim,train_data, val_X=val_data,lambda_reg=lambda_reg)
    else:
        optimized_matrices, train_loss, val_loss = train_torch(hilbert_dim,train_data, val_X=val_data,lambda_reg=lambda_reg)
    A_numpy = (tonumpy(f)(optimized_matrices)) 
    for A in A_numpy:
        print(A)
    

    plt.figure()
    plt.plot(train_loss,color='b',label='Train')
    plt.plot(val_loss,color='g',label='Val')
    plt.legend()
    plt.grid()
    plt.show()

    quantum_metrics, estimated_data, estimated_eigvals = compute_quantum_metric(A_numpy, val_data)
    print(estimated_data.shape)

    plot_manifold(val_data,estimated_data)

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