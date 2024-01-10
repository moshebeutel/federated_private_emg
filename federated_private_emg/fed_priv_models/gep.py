# Taken from https://github.com/dayu11/Gradient-Embedding-Perturbation
import gc
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# package for computing individual gradients
from federated_private_emg.common.config import Config
from federated_private_emg.common.utils import flatten_tensor


# from memory_profiler import profile

@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # print(f'Normalize the {i}-th column')
        #
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


def clip_column(tsr, clip=1.0, inplace=True):
    if inplace:
        if torch.cuda.is_available() and Config.DEVICE == 'cuda':
            inplace_clipping(tsr, torch.tensor(clip).cuda())
        else:
            inplace_clipping(tsr, torch.tensor(clip))
    else:
        norms = torch.norm(tsr, dim=1)

        scale = torch.clamp(clip / norms, max=1.0)
        return tsr * scale.view(-1, 1)


@torch.jit.script
def inplace_clipping(matrix, clip):
    n, m = matrix.shape
    for i in range(n):
        # Normalize the i'th row
        col = matrix[i:i + 1, :]
        col_norm = torch.sqrt(torch.sum(col ** 2))
        if (col_norm > clip):
            col /= (col_norm / clip)


def check_approx_error_pca(pca: PCA, target: np.array) -> float:
    """
        Calculate the normalized approximation error between the target matrix and its PCA reconstruction.

        Parameters:
        - pca (PCA): The PCA (Principal Component Analysis) model fitted on the training data.
        - target (np.ndarray): The target matrix of shape (n, m) where
                                    n is the number of samples
                                        and
                                    m is the number of features.

        Returns:
        - float: The normalized approximation error between the target matrix and its PCA reconstruction.

        Raises:
        - AssertionError: If the L2 norm of the target matrix is zero, indicating a zero vector.

        Example:
        ```python
        from sklearn.decomposition import PCA
        import numpy as np

        # Assuming pca is a fitted PCA model
        pca = PCA(n_components=3)
        pca.fit(training_data)

        target = np.random.rand(20, 10)
        error = check_approx_error_pca(pca, target)
        print(f"PCA Approximation error: {error}")


        Note:
        The function uses a pre-fitted PCA model to compute the PCA embedding and reconstruction of the target matrix.
        The approximation error is then calculated as the L2 norm of the difference between the target matrix and
        its PCA reconstruction,
        normalized by the L2 norm of the target matrix.
        """
    target_norm: float = np.linalg.norm(target)
    assert target_norm != 0, 'target is zero vector'

    embedding: np.ndarray = pca.transform(target)
    approx: np.ndarray = pca.inverse_transform(embedding)
    k = np.diag(cosine_similarity(target, approx))
    norm_similarity = np.mean(np.abs(k), axis=0)
    return 1.0 - norm_similarity


def check_embedding_error(grad: torch.tensor, basis: torch.tensor) -> float:
    target = torch.mean(grad, dim=0)
    target_norm = float(torch.norm(target))
    assert target_norm != 0, 'target is zero vector'

    embedding = torch.matmul(grad, basis)
    approx = torch.matmul(torch.mean(embedding, dim=0).view(1, -1), basis.T).view(-1)
    target_norm = float(torch.norm(target))
    error = float(torch.norm(approx - target))
    embedding_error: float = error / target_norm
    return embedding_error


def check_approx_error(L, target) -> float:
    target_norm = float(torch.norm(target))
    assert target_norm != 0, 'target is zero vector'

    encode = torch.matmul(target, L)  # n x k
    decode = torch.matmul(encode, L.T)
    error = float(torch.norm(target - decode))
    return error / target_norm


def check_approx_error_np(L: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate the normalized approximation error between the target matrix and its reconstruction.

    Parameters:
    - L (np.ndarray): The encoding matrix of shape (m, k) where
                        m is the number of features
                            and
                        k is the number of components.
    - target (np.ndarray): The target matrix of shape (n, m) where
                        n is the number of samples
                            and
                        m is the number of features.

    Returns:
    - float: The normalized approximation error between the target matrix and its reconstruction using L.

    Raises:
    - AssertionError: If the L2 norm of the target matrix is zero, indicating a zero vector.

    Example:
    ```python
    L = np.random.rand(10, 5)
    target = np.random.rand(20, 10)
    error = check_approx_error_np(L, target)
    print(f"Approximation error: {error}")
    ```
    Note:
    The function computes the encoding and decoding of the target matrix using the encoding matrix L.
    The approximation error is then calculated as the L2 norm of the difference between
     the target matrix and its reconstruction,
    normalized by the L2 norm of the target matrix.
    """
    target_norm: float = np.linalg.norm(target)
    assert target_norm != 0, 'target is zero vector'

    target_mean: np.array = target.mean(0, keepdims=True)  # (n,1)
    target_zero_mean: np.array = target - target_mean  # (n, m)

    encode: np.array = np.matmul(target_zero_mean, L)  # (n, m) X (m, k) = (n, k)
    decode: np.array = np.matmul(encode, L.T) + target_mean  # (n, k) X (k, m) = (n, m)
    # decode_mean_shift_back: np.array = decode + target_mean  # (n, m) + (n,1) broadcasts to (n, m)
    error: float = np.linalg.norm(target - decode)

    return error / target_norm


# @profile
def get_bases(pub_grad: np.array, num_bases: int):
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]
    num_bases = min(num_bases, min(num_p, num_k))
    pca = PCA(n_components=num_bases)
    pca.fit(pub_grad)

    return pca


def log_embedding_norms(concatenated_embedding_pca, residual_gradients_pca):
    norms_pca = torch.norm(concatenated_embedding_pca, dim=1)
    mean_embedding = torch.mean(norms_pca).item()
    max_embedding = torch.max(norms_pca).item()
    median_embedding = torch.median(norms_pca).item()
    norms_pca = torch.norm(residual_gradients_pca, dim=1)
    mean_residual = torch.mean(norms_pca).item()
    max_residual = torch.max(norms_pca).item()
    median_residual = torch.median(norms_pca).item()
    print('*** max norm ***')
    print('residual', max_residual)
    print('embedding', max_embedding)
    print('ratio', max_residual / max_embedding)
    print('*** mean norm ***')
    print('residual', mean_residual)
    print('embedding', mean_embedding)
    print('ratio', mean_residual / mean_embedding)
    print('*** median norm ***')
    print('residual', median_residual)
    print('embedding', median_embedding)
    print('ratio', median_residual / median_embedding)
    print('*** shape ***')
    print('residual', residual_gradients_pca.shape)
    print('embedding', concatenated_embedding_pca.shape)


def get_approx_grad(embedding, bases_list, num_bases_list):
    grad_list = []
    offset = 0
    if len(embedding.shape) > 1:
        bs = embedding.shape[0]
    else:
        bs = 1
    embedding = embedding.view(bs, -1)

    for i, bases in enumerate(bases_list):
        num_bases = num_bases_list[i]

        bases_components = bases.T

        grad = torch.matmul(embedding[:, offset:offset + num_bases].view(bs, -1), bases_components)

        if bs > 1:
            grad_list.append(grad.view(bs, -1))
        else:
            grad_list.append(grad.view(-1))
        offset += num_bases
    if bs > 1:
        return torch.cat(grad_list, dim=1)
    else:
        return torch.cat(grad_list)


class GEP(nn.Module):

    def __init__(self, num_bases, batch_size, clip0=1, clip1=1, power_iter=1):
        super(GEP, self).__init__()

        self._last_pub_grad: torch.tensor = None
        self.num_bases_list: list[int] = []
        self.num_bases: int = num_bases
        self.clip0: float = clip0
        self.clip1: float = clip1
        self.power_iter = power_iter
        self.batch_size = batch_size

        self.public_approx_error = []
        self.public_approx_error_pca = []
        self.public_diff_approx_error = []

        self.private_approx_error = []
        self.private_approx_error_pca = []
        self.private_approx_error_noised_pca = []
        self.private_diff_approx_error = []

        self.max_grads = Config.GEP_HISTORY_GRADS
        self._selected_bases_list = None
        # self._selected_pca_list: list[torch.tensor] = [None for _ in range(Config.GEP_NUM_GROUPS)]
        self.history_anchor_grads = None
        self._max_norm_grad = 0
        self._pca = PCA(n_components=num_bases)
        self._noised_pca = PCA(n_components=num_bases)
        self._anchor_grads = None
        self._noised_anchor_grads = None

        self.iter = 0

    @property
    def max_norm_grad(self):
        return self._max_norm_grad

    # @property
    # def selected_bases_list(self):
    #     return self._selected_pca_list if Config.GEP_USE_PCA else self._selected_bases_list

    def get_anchor_gradients(self, net, loss_func):
        cur_batch_grad_list = []
        for p in net.parameters():
            if hasattr(p, 'grad_batch'):
                grad_batch = p.grad_batch[:len(self.public_users)]
                cur_batch_grad_list.append(grad_batch.reshape(grad_batch.shape[0], -1))
        return flatten_tensor(cur_batch_grad_list)

    def append_current_anchor_grads_keep_history_size(self, current_anchor_grads):
        self.history_anchor_grads = torch.cat([self.history_anchor_grads, current_anchor_grads]) \
            if self.history_anchor_grads is not None else current_anchor_grads
        if self.history_anchor_grads.shape[0] > self.max_grads:
            self.history_anchor_grads = self.history_anchor_grads[-self.max_grads:, :]

    # @profile
    def get_anchor_space(self, net, loss_func, logging=False):
        current_anchor_grads = self.get_anchor_gradients(net, loss_func)
        anchor_grads: torch.tensor
        if not Config.SANITY_CHECK:
            self.append_current_anchor_grads_keep_history_size(current_anchor_grads=current_anchor_grads)
            anchor_grads = self.history_anchor_grads
        else:
            anchor_grads = current_anchor_grads
        noise_for_anchor_grads: torch.tensor = torch.normal(0, Config.GEP_SIGMA0 * Config.GEP_CLIP0,
                                   size=anchor_grads.shape,
                                   device=anchor_grads.device) / Config.NUM_CLIENT_AGG
        self._anchor_grads = anchor_grads
        self._noised_anchor_grads = anchor_grads + noise_for_anchor_grads
        print(self._anchor_grads.shape)

        with (torch.no_grad()):
            num_param_list = self.num_param_list

            # selected_bases_list = []
            # selected_pca_list: torch.tensor = []
            pub_approx_errs_list: list[float] = []
            pub_approx_pca_errs_list: list[float] = []
            pub_diff_approx_errs_list: list[float] = []

            sqrt_num_param_list = np.sqrt(np.array(num_param_list))
            # *** Cancel normalization to avoid all zeros when casting to int in TOY_STORY
            num_bases_list = self.num_bases * (sqrt_num_param_list / np.sum(sqrt_num_param_list))
            num_bases_list = num_bases_list.astype(int)

            offset = 0
            device = next(net.parameters()).device
            for i, num_param in enumerate(num_param_list):
                pub_grad: torch.tensor = anchor_grads[:, offset:offset + num_param]
                noised_pub_grad: torch.tensor = self._noised_anchor_grads[:, offset:offset + num_param]

                offset += num_param

                num_bases = num_bases_list[i]

                # print("PUBLIC GET BASES")
                # self._last_pub_grad = pub_grad
                # mean = pub_grad.mean(0, keepdim=True)
                # print('pub grad mean largest value', mean.absolute().max())
                # num_bases, pub_error, pca = get_bases(pub_grad - mean, num_bases, self.power_iter, logging)
                pub_grad_np: np.array = pub_grad.cpu().detach().numpy()
                noised_pub_grad_np: np.array = noised_pub_grad.cpu().detach().numpy()
                pca: PCA = get_bases(pub_grad_np, num_bases)
                noised_pca: PCA = get_bases(noised_pub_grad_np, num_bases)

                # with open(f'pub_grad_iter_{self.iter}.npy', 'wb') as f:
                #     np.save(f, pub_grad_np)

                # with open(f'pca_components_iter_{self.iter}.npy', 'wb') as f:
                #     np.save(f, pca.components_)
                # joblib.dump(pca, f'pca_iter_{self.iter}.pkl')

                # pub_error: float = check_approx_error(torch.from_numpy(pca.components_).T.to(pub_grad.device),
                #                                        pub_grad)
                pub_approx_error: float = check_approx_error_np(pca.components_.T, pub_grad_np)
                pub_approx_error_pca: float = check_approx_error_pca(pca, pub_grad_np)
                diff_errors: float = abs(pub_approx_error - pub_approx_error_pca)
                pub_approx_errs_list.append(pub_approx_error)
                pub_approx_pca_errs_list.append(pub_approx_error_pca)
                pub_diff_approx_errs_list.append(diff_errors)
                # selected_pca: torch.tensor = torch.from_numpy(pca.components_).squeeze().T.to(device)
                # cur_error_pca: float = check_embedding_error(pub_grad - mean, selected_pca)
                # cur_error_pca: float = check_embedding_error(pub_grad, selected_pca)
                self._pca = pca
                self._noised_pca = noised_pca
                # print('PUBLIC: Embedding error pca: %.2f%%' % (100 * cur_error_pca))
                # print('PUBLIC: Approximate error pca: %.2f%%' % (100 * pub_error))
                num_bases_list[i] = num_bases
                # selected_bases_list.append(selected_bases.T.to(device))
                # selected_pca_list.append(selected_pca)
                del pub_grad, pub_grad_np, pca, noised_pca, noised_pub_grad, noised_pub_grad_np
                # del pub_grad, pub_error, pca

            # self.selected_bases_list = selected_bases_list
            # self._selected_pca_list = selected_pca_list
            self.num_bases_list = num_bases_list
            self.public_approx_error = pub_approx_errs_list
            self.public_approx_error_pca = pub_approx_pca_errs_list
            self.public_diff_approx_error = pub_diff_approx_errs_list
            del pub_approx_errs_list, num_bases_list, pub_approx_pca_errs_list, pub_diff_approx_errs_list
            # del pub_errs, num_bases_list, selected_bases_list
        del anchor_grads
        gc.collect()

    def get_inverse_proj(self, t: torch.tensor):
        return torch.from_numpy(self._pca.inverse_transform(t.cpu().detach().numpy())).to(t.device)

    # @profile
    def forward(self, target_grad, logging=True):
        assert (not Config.SANITY_CHECK) or torch.allclose(target_grad, self._anchor_grads)
        with torch.no_grad():
            num_param_list = self.num_param_list
            # embedding_list = []
            embedding_by_pca_list = []

            offset = 0

            reconstruction_errs = []
            pca_reconstruction_errs = []
            noised_pca_reconstruction_errs = []
            diff_approx_errs_list = []
            max_norm_grad: float = 0.0
            for i, num_param in enumerate(num_param_list):
                grad = target_grad[:, offset:offset + num_param]
                max_norm_grad = max(max_norm_grad, float(torch.max(torch.norm(grad))))
                # selected_pca = self._selected_pca_list[i].to(grad.device)
                grad_np = grad.cpu().detach().numpy()
                # with open(f'private_grad_iter_{self.iter}.npy', 'wb') as f:
                #     np.save(f, grad_np)
                embedding_np = self._pca.transform(grad_np)
                embedding_by_pca = torch.from_numpy(embedding_np).to(grad.device)

                # with open(f'embedding_np_iter_{self.iter}.npy', 'wb') as f:
                #     np.save(f, embedding_np)
                #
                # self.iter += 1
                # if self.iter > 4:
                #     raise Exception

                approx_error: float = check_approx_error_np(self._pca.components_.T, grad_np)
                approx_error_pca: float = check_approx_error_pca(self._pca, grad_np)
                approx_error_noised_pca: float = check_approx_error_pca(self._noised_pca, grad_np)
                diff_errors: float = abs(approx_error - approx_error_pca)
                reconstruction_errs.append(approx_error)
                pca_reconstruction_errs.append(approx_error_pca)
                noised_pca_reconstruction_errs.append(approx_error_noised_pca)
                diff_approx_errs_list.append(diff_errors)
                # mean = grad.mean(1, keepdim=True)
                # embedding_by_pca = torch.matmul(grad - mean, selected_pca)
                # embedding_by_pca += mean
                # pca_reconstruction_error = check_approx_error(selected_pca, grad - mean)
                # pca_reconstruction_error = check_approx_error_np(self._pca.components_.T, grad_np)
                # pca_reconstruction_errs.append(pca_reconstruction_error)
                #
                # if logging:
                #     print('PRIVATE: Approximate error pca: %.2f%%' % (100 * pca_reconstruction_error))
                #     cur_error_pca: float = self.check_embedding_error(grad, selected_pca)
                #     print('PRIVATE: Embedding error pca: %.2f%%' % (100 * cur_error_pca))
                #     if i in self.approx_error_pca:
                #         self.approx_error_pca[i].append(cur_error_pca)
                #     else:
                #         self.approx_error_pca[i] = []
                #         self.approx_error_pca[i].append(cur_error_pca)

                # embedding_list.append(embedding)
                embedding_by_pca_list.append(embedding_by_pca)
                offset += num_param
                del grad, embedding_by_pca, grad_np
            self.private_approx_error = reconstruction_errs
            self.private_approx_error_pca = pca_reconstruction_errs
            self.private_approx_error_noised_pca = noised_pca_reconstruction_errs
            self.private_diff_approx_error = diff_approx_errs_list

            self._max_norm_grad = max_norm_grad
            concatenated_embedding_pca = torch.cat(embedding_by_pca_list, dim=1)

            clipped_embedding_pca = clip_column(concatenated_embedding_pca, clip=self.clip0, inplace=False)

            # avg_clipped_embedding = torch.sum(clipped_embedding, dim=0) / self.batch_size
            avg_clipped_embedding_pca = torch.sum(clipped_embedding_pca, dim=0) / self.batch_size
            del clipped_embedding_pca

            # no_reduction_approx_pca = get_approx_grad(concatenated_embedding_pca, bases_list=self._selected_pca_list,
            #                                           num_bases_list=self.num_bases_list)
            no_reduction_approx_pca = self.get_inverse_proj(concatenated_embedding_pca)

            # if Config.GEP_HISTORY_GRADS > 0:
            #     min_shape = min(target_grad.shape[1], no_reduction_approx_pca.shape[1])
            #     residual_gradients_pca = target_grad[:, :min_shape] - no_reduction_approx_pca[:, :min_shape]
            # else:
            #     residual_gradients_pca = target_grad - no_reduction_approx_pca

            residual_gradients_pca = target_grad - no_reduction_approx_pca

            clip_column(residual_gradients_pca, clip=self.clip1)  # inplace clipping to save memory
            clipped_residual_gradients_pca = residual_gradients_pca

            avg_clipped_residual_gradients_pca = torch.sum(clipped_residual_gradients_pca, dim=0) / self.batch_size
            avg_target_grad = torch.sum(target_grad, dim=0) / self.batch_size
            del no_reduction_approx_pca, residual_gradients_pca, clipped_residual_gradients_pca

            gc.collect()

            if Config.GEP_USE_PCA:
                return avg_clipped_embedding_pca.view(-1), avg_clipped_residual_gradients_pca.view(
                    -1), avg_target_grad.view(-1)
            else:
                pass
                # return avg_clipped_embedding.view(-1), None, avg_target_grad.view(-1)
                # return avg_clipped_embedding.view(-1), avg_clipped_residual_gradients.view(-1), avg_target_grad.view(-1)
