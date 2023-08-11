# Taken from https://github.com/dayu11/Gradient-Embedding-Perturbation
import gc

import numpy as np
import torch
import torch.nn as nn
# package for computing individual gradients
from backpack import backpack
from backpack.extensions import BatchGrad

from common.config import Config
from common.utils import flatten_tensor


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


def check_approx_error(L, target):
    encode = torch.matmul(target, L)  # n x k
    decode = torch.matmul(encode, L.T)
    error = torch.sum(torch.square(target - decode))
    target = torch.sum(torch.square(target))
    if (target.item() == 0):
        return -1
    return error.item() / target.item()


def get_bases(pub_grad, num_bases, power_iter=1, logging=False):
    from sklearn.decomposition import PCA
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]

    num_bases = min(num_bases, num_p)
    # print(pub_grad.shape[1], num_bases)
    L = torch.normal(0, 1.0, size=(pub_grad.shape[1], num_bases), device=pub_grad.device)
    for i in range(power_iter):
        # print('get bases: iter', i)
        R = torch.matmul(pub_grad, L)  # n x k
        L = torch.matmul(pub_grad.T, R)  # p x k
        # print(R,L)
        orthogonalize(L)
    error_rate = check_approx_error(L, pub_grad)

    pca = PCA(n_components=num_bases)
    pca.fit(pub_grad.detach().numpy())
    # print(pca.explained_variance_)
    # print(pca.explained_variance_ratio_)
    # print(pca.explained_variance_.sum())
    # print(pca.explained_variance_.cumsum())
    # print(pca.explained_variance_ratio_.sum())
    # print(pca.explained_variance_ratio_.cumsum())

    return L, num_bases, error_rate, pca


class GEP(nn.Module):

    def __init__(self, num_bases, batch_size, clip0=1, clip1=1, power_iter=1):
        super(GEP, self).__init__()

        self.selected_pca_list = None
        self.selected_bases_list = None
        self.num_bases = num_bases
        self.clip0 = clip0
        self.clip1 = clip1
        self.power_iter = power_iter
        self.batch_size = batch_size
        self.approx_error = {}
        self.approx_error_pca = {}
        self.base_history = Config.GEP_HISTORY_GRADS
        self.selected_bases_list_history_list = []

    def get_approx_grad(self, embedding):
        bases_list, num_bases_list, num_param_list = self.selected_bases_list, self.num_bases_list, self.num_param_list
        grad_list = []
        offset = 0
        if len(embedding.shape) > 1:
            bs = embedding.shape[0]
        else:
            bs = 1
        embedding = embedding.view(bs, -1)

        for i, bases in enumerate(bases_list):
            num_bases = num_bases_list[i]

            grad = torch.matmul(embedding[:, offset:offset + num_bases].view(bs, -1), bases.T)
            if bs > 1:
                grad_list.append(grad.view(bs, -1))
            else:
                grad_list.append(grad.view(-1))
            offset += num_bases
        if bs > 1:
            return torch.cat(grad_list, dim=1)
        else:
            return torch.cat(grad_list)

    def get_anchor_gradients(self, net, loss_func):
        print('get_anchor_gradient')
        # device = next(net.parameters()).device
        # for inputs, targets in self.loader:
        #     inputs, targets = inputs.to(device), targets.to(device)
        #     outputs = net(inputs)
        #     loss = loss_func(outputs.cpu(), targets.cpu())
        #     with backpack(BatchGrad()):
        #         # print('get_anchor_gradients. before loss.backward()')
        #         loss.backward()
        #     del loss, outputs
        cur_batch_grad_list = []
        for p in net.parameters():
            # print('get_anchor_gradients. p.grad_batch.shape', p.grad_batch.shape)
            if hasattr(p, 'grad_batch'):
                grad_batch = p.grad_batch[:len(self.public_users)]
                cur_batch_grad_list.append(grad_batch.reshape(grad_batch.shape[0], -1))
                # del p.grad_batch

        gc.collect()

        return flatten_tensor(cur_batch_grad_list)

    # @profile
    def get_anchor_space(self, net, loss_func, logging=False):
        # print('get_anchor_space')
        anchor_grads = self.get_anchor_gradients(net, loss_func)  # \
        # if self.selected_bases_list \
        # else torch.ones((self.batch_size, self.num_params))
        # print('get_anchor_space. anchor_grads.shape ', anchor_grads.shape)
        with torch.no_grad():
            num_param_list = self.num_param_list

            selected_bases_list = []
            selected_pca_list = []
            pub_errs = []

            sqrt_num_param_list = np.sqrt(np.array(num_param_list))
            # *** Cancel normalization to avoid all zeros when casting to int in TOY_STORY
            num_bases_list = self.num_bases * (sqrt_num_param_list / np.sum(sqrt_num_param_list))
            # num_bases_list = self.num_bases * sqrt_num_param_list
            num_bases_list = num_bases_list.astype(int)

            offset = 0
            device = next(net.parameters()).device
            for i, num_param in enumerate(num_param_list):
                pub_grad = anchor_grads[:, offset:offset + num_param]
                # print(f'i = {i}, num_param = {num_param}. pub_grad.shape {pub_grad.shape}')
                offset += num_param

                num_bases = num_bases_list[i]
                # print('before get bases. num_bases', num_bases)
                selected_bases, num_bases, pub_error, pca = get_bases(pub_grad, num_bases, self.power_iter, logging)
                pub_errs.append(pub_error)

                num_bases_list[i] = num_bases
                selected_bases_list.append(selected_bases.to(device))
                selected_pca_list.append(pca)
                del pub_grad, pub_error

            if Config.GEP_HISTORY_GRADS > 0:
                min_shape = selected_bases_list[0].shape
                self.selected_bases_list_history_list.append(
                torch.stack([base[:min_shape[0], :min_shape[1]] for base in selected_bases_list]))
                tmp = torch.stack(self.selected_bases_list_history_list)
                self.selected_bases_list = tmp.mean(dim=0)
                if len(self.selected_bases_list_history_list) >= self.base_history:
                    self.selected_bases_list_history_list = self.selected_bases_list_history_list[1:]
            else:
                self.selected_bases_list = selected_bases_list
                self.selected_pca_list = selected_pca_list

            print(f'selected bases history list len {len(self.selected_bases_list_history_list)}')
            print(f'selected bases list len {len(self.selected_bases_list)}')
            # print('self.selected_bases_list', self.selected_bases_list)
            self.num_bases_list = num_bases_list
            self.approx_errors = pub_errs
            del pub_errs, num_bases_list, selected_bases_list
            gc.collect()
        del anchor_grads
        gc.collect()

    def forward(self, target_grad, logging=True):
        # print('gep forward')
        with torch.no_grad():
            num_param_list = self.num_param_list
            embedding_list = []
            embedding_by_pca_list = []

            offset = 0
            if logging:
                print('group wise approx error')
            reconstruction_errs = []
            pca_reconstruction_errs = []
            for i, num_param in enumerate(num_param_list):
                grad = target_grad[:, offset:offset + num_param]
                # print(f'i = {i}, num_param = {num_param}. grad.shape {grad.shape}')
                # print(f'selected bases list len {len(self.selected_bases_list)}')
                selected_bases = self.selected_bases_list[i]
                selected_pca = self.selected_pca_list[i]
                # print(selected_bases.shape)
                if Config.GEP_HISTORY_GRADS > 0:
                    min_shape = min(grad.shape[1], selected_bases.shape[0])
                    embedding = torch.matmul(grad[:, :min_shape], selected_bases[:min_shape, :])
                else:
                    embedding = torch.matmul(grad, selected_bases)
                    selected_pca_components_tensor = torch.from_numpy(selected_pca.components_.T)
                    embedding_by_pca = torch.matmul(grad, selected_pca_components_tensor)
                    bases_tensor = torch.cat([selected_bases])
                    bases_reconstruction_error = check_approx_error(bases_tensor, grad)
                    pca_reconstruction_error = check_approx_error(selected_pca_components_tensor, grad)
                    print('bases_reconstruction_error', bases_reconstruction_error)
                    print('pca_reconstruction_error', pca_reconstruction_error)
                    reconstruction_errs.append(bases_reconstruction_error)
                    pca_reconstruction_errs.append(pca_reconstruction_error)

                num_bases = self.num_bases_list[i]
                if logging:
                    cur_approx = torch.matmul(torch.mean(embedding, dim=0).view(1, -1), selected_bases.T).view(-1)
                    cur_approx_pca = torch.matmul(torch.mean(embedding_by_pca, dim=0).view(1, -1),
                                                  selected_pca_components_tensor.T).view(-1)
                    cur_target = torch.mean(grad, dim=0)
                    cur_target_sqr_norm = torch.sum(torch.square(cur_target))
                    cur_error = torch.sum(torch.square(cur_approx - cur_target)) / cur_target_sqr_norm
                    print('group %d, param: %d, num of bases: %d, group wise approx error: %.2f%%' % (
                        i, num_param, self.num_bases_list[i], 100 * cur_error.item()))

                    cur_error_pca = torch.sum(torch.square(cur_approx_pca - cur_target)) / cur_target_sqr_norm
                    print('group wise approx error pca: %.2f%%' % (100 * cur_error_pca.item()))

                    if i in self.approx_error:
                        self.approx_error[i].append(cur_error.item())
                    else:
                        self.approx_error[i] = []
                        self.approx_error[i].append(cur_error.item())

                    if i in self.approx_error_pca:
                        self.approx_error_pca[i].append(cur_error_pca.item())
                    else:
                        self.approx_error_pca[i] = []
                        self.approx_error_pca[i].append(cur_error_pca.item())

                embedding_list.append(embedding)
                embedding_by_pca_list.append(embedding_by_pca)
                offset += num_param
                del grad, embedding, selected_bases, embedding_by_pca, selected_pca

            concatnated_embedding = torch.cat(embedding_list, dim=1)
            concatnated_embedding_pca = torch.cat(embedding_by_pca_list, dim=1)
            clipped_embedding = clip_column(concatnated_embedding, clip=self.clip0, inplace=False)
            clipped_embedding_pca = clip_column(concatnated_embedding_pca, clip=self.clip0, inplace=False)
            if logging:
                norms = torch.norm(clipped_embedding, dim=1)
                norms_pca = torch.norm(clipped_embedding_pca, dim=1)
                print('power: average norm of clipped embedding: ', torch.mean(norms).item(), 'max norm: ',
                      torch.max(norms).item(), 'median norm: ', torch.median(norms).item())
                print('pca: average norm of clipped embedding: ', torch.mean(norms_pca).item(), 'max norm: ',
                      torch.max(norms_pca).item(), 'median norm: ', torch.median(norms_pca).item())
            avg_clipped_embedding = torch.sum(clipped_embedding, dim=0) / self.batch_size
            avg_clipped_embedding_pca = torch.sum(clipped_embedding_pca, dim=0) / self.batch_size
            del clipped_embedding, clipped_embedding_pca

            no_reduction_approx = self.get_approx_grad(concatnated_embedding)
            no_reduction_approx_pca = self.get_approx_grad(concatnated_embedding_pca)
            if Config.GEP_HISTORY_GRADS > 0:
                min_shape = min(target_grad.shape[1], no_reduction_approx.shape[1])
                residual_gradients = target_grad[:, :min_shape] - no_reduction_approx[:, :min_shape]
            else:
                residual_gradients = target_grad - no_reduction_approx
                residual_gradients_pca = target_grad - no_reduction_approx_pca

            clip_column(residual_gradients, clip=self.clip1)  # inplace clipping to save memory
            clip_column(residual_gradients_pca, clip=self.clip1)  # inplace clipping to save memory
            clipped_residual_gradients = residual_gradients
            clipped_residual_gradients_pca = residual_gradients_pca
            if logging:
                norms = torch.norm(clipped_residual_gradients, dim=1)
                norms_pca = torch.norm(clipped_residual_gradients_pca, dim=1)
                print('power: average norm of clipped residual gradients: ', torch.mean(norms).item(), 'max norm: ',
                      torch.max(norms).item(), 'median norm: ', torch.median(norms).item())
                print('pca: average norm of clipped residual gradients: ', torch.mean(norms_pca).item(), 'max norm: ',
                      torch.max(norms_pca).item(), 'median norm: ', torch.median(norms_pca).item())

            avg_clipped_residual_gradients = torch.sum(clipped_residual_gradients, dim=0) / self.batch_size
            avg_clipped_residual_gradients_pca = torch.sum(clipped_residual_gradients_pca, dim=0) / self.batch_size
            avg_target_grad = torch.sum(target_grad, dim=0) / self.batch_size
            del no_reduction_approx, residual_gradients, clipped_residual_gradients, no_reduction_approx_pca,\
                residual_gradients_pca, clipped_residual_gradients_pca

            if Config.GEP_USE_PCA:
                return avg_clipped_embedding_pca.view(-1), avg_clipped_residual_gradients_pca.view(-1), avg_target_grad.view(-1)
            else:
                return avg_clipped_embedding.view(-1), avg_clipped_residual_gradients.view(-1), avg_target_grad.view(-1)
