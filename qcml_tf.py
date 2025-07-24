import tensorflow as tf
import numpy as np

import tensorflow as tf

class HermitianMatrixLayer(tf.keras.layers.Layer):
    def __init__(self, dim, num_matrices, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_matrices = num_matrices

        # Real diagonal
        self.diagonal = self.add_weight(
            shape=(num_matrices, dim),
            initializer="random_normal",
            trainable=True,
            dtype=tf.float32,
            name="diagonal"
        )

        # Real part of upper triangle (excluding diagonal)
        self.upper_triangle_real = self.add_weight(
            shape=(num_matrices, dim, dim),
            initializer="random_normal",
            trainable=True,
            dtype=tf.float32,
            name="upper_triangle_real"
        )

        # Imag part of upper triangle (excluding diagonal)
        self.upper_triangle_imag = self.add_weight(
            shape=(num_matrices, dim, dim),
            initializer="random_normal",
            trainable=True,
            dtype=tf.float32,
            name="upper_triangle_imag"
        )

    def call(self, inputs=None):  # no inputs expected
        matrices = []
        for i in range(self.num_matrices):
            diag = tf.complex(tf.linalg.diag(self.diagonal[i]), 0.0)

            upper_real = tf.linalg.band_part(self.upper_triangle_real[i], 0, -1) - tf.linalg.diag(tf.linalg.diag_part(self.upper_triangle_real[i]))
            upper_imag = tf.linalg.band_part(self.upper_triangle_imag[i], 0, -1) - tf.linalg.diag(tf.linalg.diag_part(self.upper_triangle_imag[i]))

            upper = tf.complex(upper_real, upper_imag)
            hermitian = diag + upper + tf.transpose(upper, perm=[1, 0], conjugate=True)
            matrices.append(hermitian)
        return tf.stack(matrices)  # shape: (num_matrices, dim, dim)

def calc_position_and_variance(A, X):

    N = tf.shape(X)[0]
    D = tf.shape(X)[1]
    dim = tf.shape(A)[1]

    X_real = tf.cast(X, dtype=tf.float32)
    valid = tf.math.is_finite(X_real)
    X_tensor = tf.cast(X_real, tf.complex64) # shape (N, D)
    I = tf.eye(dim, dtype=tf.complex64)
    I_exp = tf.reshape(I, (1, 1, dim, dim))
    I_exp = tf.tile(I_exp, [N, D, 1, 1])  # shape (N, D, dim, dim)
    
    X_cmplx = tf.reshape(X_tensor, (N, D, 1, 1))
    scaled_I = X_cmplx * I_exp

    valid_exp = tf.reshape(valid, (N, D, 1, 1))
    A_exp = tf.cast(tf.tile(tf.expand_dims(A, axis=0), [N, 1, 1, 1]), tf.complex64) # shape (N, D, dim, dim)

    scaled_I_clean = tf.where(valid_exp, scaled_I, A_exp)
    H_diff = A_exp - scaled_I_clean
    H_diff = H_diff * tf.cast(valid_exp, tf.complex64)

    H_x = tf.einsum('ndij,ndkj->ndik', H_diff, tf.math.conj(H_diff))
    H_x_final = 0.5 * tf.reduce_sum(H_x, axis=1)  # shape (N, dim, dim)
    eps = tf.constant(1e-5, dtype=tf.complex64)
    H_x_final += eps * tf.eye(dim, dtype=tf.complex64)

    eigvals, eigvecs = tf.linalg.eigh(H_x_final)  # eigvecs: (N, dim, dim)
    psi_0 = eigvecs[:, :, 0]  # (N, dim)

    # Construct projector
    psi_0_col = tf.expand_dims(psi_0, -1)  # (N, dim, 1)
    psi_0_adj = tf.transpose(psi_0_col, perm=[0, 2, 1], conjugate=True)  # (N, 1, dim)
    projector = tf.matmul(psi_0_col, psi_0_adj)  # (N, dim, dim)

    A_tensor_exp = tf.cast(tf.tile(tf.expand_dims(A, 0), [N, 1, 1, 1]), tf.complex64)  # (N, D, dim, dim)
    assert tf.reduce_max(tf.abs(A_tensor_exp - tf.linalg.adjoint(A_tensor_exp))) < 1e-5
    assert tf.reduce_max(tf.abs(projector - tf.linalg.adjoint(projector))) < 1e-5        
    position_vector = tf.einsum('ndij,njk->nd', A_tensor_exp, projector)  # (N, D)
    
    A2_tensor_exp = tf.matmul(A_tensor_exp, A_tensor_exp)  # (N, D, dim, dim)
    expectation_A2 = tf.einsum('ndij,njk->nd', A2_tensor_exp, projector)  # (N, D)
    position_variance = tf.reduce_sum(expectation_A2 - position_vector ** 2)
    return position_vector, position_variance 

class EnergyLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_reg=0.1):
        super(EnergyLoss, self).__init__()
        self.lambda_reg = lambda_reg
   
    def call(self,A,X):
        if isinstance(X, np.ndarray):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        position_vector,position_variance = calc_position_and_variance(A,X)
        X_tensor = tf.cast(X, dtype=tf.complex64)
        N = tf.shape(X)[0]
        position_error = tf.reduce_sum((position_vector - X_tensor) ** 2)
        loss = tf.math.real(position_error + self.lambda_reg * position_variance)
        return loss / tf.cast(N, tf.float32)

def train_model(dim, X, val_X=None, max_iter=400, learning_rate=0.01, patience=10, min_delta=1e-5, lambda_reg=0.1):
    num_matrices = X.shape[1]
    layer = HermitianMatrixLayer(dim, num_matrices)
    loss_fn = EnergyLoss(lambda_reg=lambda_reg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_trajectory = []
    val_loss_trajectory = []
    best_weights = None
    best_loss = np.inf
    patience_counter = 0

    for epoch in range(max_iter):
        with tf.GradientTape() as tape:
            matrices = layer()
            loss = loss_fn(matrices, X)
        grads = tape.gradient(loss, layer.trainable_variables)
        optimizer.apply_gradients(zip(grads, layer.trainable_variables))

        loss_val = loss.numpy()
        loss_trajectory.append(loss_val)

        if val_X is not None:
            val_loss = loss_fn(matrices, val_X).numpy()
            val_loss_trajectory.append(val_loss)
        else:
            val_loss = loss_val

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_weights = [tf.identity(w) for w in layer.trainable_variables]
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch%50==0:
            print(f'Epoch {epoch}: loss = {loss_val}, val_loss = {val_loss}')
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best_loss:.6f}")
            break

    if best_weights is not None:
        for var, best_val in zip(layer.trainable_variables, best_weights):
            var.assign(best_val)

    final_matrices = layer()
    if val_X is not None:
        return final_matrices, loss_trajectory, val_loss_trajectory
    return final_matrices, loss_trajectory

def tf_to_numpy(data):
    """
    Recursively converts TensorFlow tensors (or nested structures) to NumPy arrays.
    """
    if isinstance(data, tf.Tensor):
        return data.numpy()
    elif isinstance(data, (list, tuple)):
        return [tf_to_numpy(d) for d in data]
    elif isinstance(data, dict):
        return {k: tf_to_numpy(v) for k, v in data.items()}
    else:
        return data
