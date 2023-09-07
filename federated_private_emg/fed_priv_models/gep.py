# Taken from https://github.com/dayu11/Gradient-Embedding-Perturbation
import gc
from sklearn.decomposition import PCA
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
    # print(f'inside check_approx_error error = {error.item()} target = {target.item()}')
    return error.item() / target.item()


def get_bases(pub_grad, num_bases, power_iter=1, logging=False):
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]

    print('$$$$$')
    print('k, p, num_bases', num_k, num_p, num_bases)
    num_bases = min(num_bases, min(num_p, num_k))
    print('num_bases', num_bases)
    print('$$$$$')
    # L = torch.normal(0, 1.0, size=(pub_grad.shape[1], num_bases), device=pub_grad.device)
    # for i in range(power_iter):
    #     # print('get bases: iter', i)
    #     R = torch.matmul(pub_grad, L)  # n x k
    #     L = torch.matmul(pub_grad.T, R)  # p x k
    #     # print(R,L)
    #     orthogonalize(L)
    # error_rate = check_approx_error(L, pub_grad)

    pca = PCA(n_components=num_bases)
    pca.fit(pub_grad.detach().numpy())
    # print(pca.explained_variance_)
    # print(pca.explained_variance_ratio_)
    # print(pca.explained_variance_.sum())
    # print(pca.explained_variance_.cumsum())
    # print(pca.explained_variance_ratio_.sum())
    # print(pca.explained_variance_ratio_.cumsum())

    cumsums = pca.explained_variance_ratio_.cumsum()

    # list08.append(len(cumsums[cumsums < 0.8]))
    # list09.append(len(cumsums[cumsums < 0.9]))
    # list095.append(len(cumsums[cumsums < 0.95]))

    num_bases_0_8 = len(cumsums[cumsums < 0.8])
    num_bases_0_9 = len(cumsums[cumsums < 0.9])
    num_bases_0_95 = len(cumsums[cumsums < 0.95])

    print('epoch_num_bases_0_8', num_bases_0_8,
          'epoch_num_bases_0_9', num_bases_0_9,
          'epoch_num_bases_0_95', num_bases_0_95)
    error_rate = check_approx_error(torch.from_numpy(pca.components_).T, pub_grad)

    return None, num_bases, error_rate, pca
    # return L, num_bases, error_rate, pca


class GEP(nn.Module):

    def __init__(self, num_bases, batch_size, clip0=1, clip1=1, power_iter=1):
        super(GEP, self).__init__()

        self.num_bases_list = []
        self.num_bases = num_bases
        self.clip0 = clip0
        self.clip1 = clip1
        self.power_iter = power_iter
        self.batch_size = batch_size
        self.approx_error = {}
        self.approx_error_pca = {}
        self.base_history = Config.GEP_HISTORY_GRADS
        # self.selected_bases_list = [torch.empty(size=(num_bases,)) for _ in range(Config.GEP_NUM_GROUPS)]
        # self.selected_bases_list_history_list = [[] for _ in range(Config.GEP_NUM_GROUPS)]
        self.selected_pca_group_list = [None for _ in range(Config.GEP_NUM_GROUPS)]
        self.selected_pca_history_list_group_list = [[] for _ in range(Config.GEP_NUM_GROUPS)]

    def get_approx_grad(self, embedding):
        bases_list, num_bases_list, num_param_list = (self.selected_bases_list if not Config.GEP_USE_PCA else
                                                      self.selected_pca_group_list, self.num_bases_list,
                                                      self.num_param_list)
        grad_list = []
        offset = 0
        if len(embedding.shape) > 1:
            bs = embedding.shape[0]
        else:
            bs = 1
        embedding = embedding.view(bs, -1)

        for i, bases in enumerate(bases_list):
            num_bases = num_bases_list[i]

            bases_components = bases.squeeze() if Config.GEP_USE_PCA else bases.T

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

    def get_anchor_gradients(self, net, loss_func):
        # print('get_anchor_gradient')
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
        anchor_grads = self.get_anchor_gradients(net, loss_func)  # \

        with (torch.no_grad()):
            num_param_list = self.num_param_list

            # selected_bases_list = []
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
                if self.selected_pca_group_list[i] is not None:
                    pub_grad = torch.cat([self.selected_pca_group_list[i], pub_grad])

                offset += num_param

                num_bases = num_bases_list[i]

                selected_bases, num_bases, pub_error, pca = get_bases(pub_grad, num_bases, self.power_iter, logging)
                pub_errs.append(pub_error)
                print('group wise approx PUBLIC  error pca: %.2f%%' % (100 * pub_error))
                num_bases_list[i] = num_bases
                # selected_bases_list.append(selected_bases.T.to(device))
                selected_pca_list.append(torch.from_numpy(pca.components_).to(device))
                # self.selected_pca_group_list[i] = pca
                del pub_grad, pub_error

            # if Config.GEP_HISTORY_GRADS > 0:
            for i in range(len(num_param_list)):
                # self.selected_bases_list[i] = \
                # self.update_bases_list_using_history(curr_selected_bases=selected_bases_list[i],
                #                                      history_list=self.selected_bases_list_history_list[i])

                self.selected_pca_group_list[i] = \
                    self.update_bases_list_using_history(curr_selected_bases=selected_pca_list[i],
                                                         history_list=self.selected_pca_history_list_group_list[i])

                num_bases_list[i] *= len(self.selected_pca_history_list_group_list[i])

            self.num_bases_list = num_bases_list
            self.approx_errors = pub_errs
            del pub_errs, num_bases_list
            # del pub_errs, num_bases_list, selected_bases_list
            gc.collect()
        del anchor_grads
        gc.collect()

    def update_bases_list_using_history(self, curr_selected_bases, history_list):
        min_shape = curr_selected_bases.shape
        history_list.append(curr_selected_bases)
        if len(history_list) > self.base_history:
            history_list = history_list[1:]
        return torch.cat(history_list)

    def forward(self, target_grad, logging=True):
        with torch.no_grad():
            num_param_list = self.num_param_list
            embedding_list = []
            embedding_by_pca_list = []

            offset = 0

            reconstruction_errs = []
            pca_reconstruction_errs = []
            for i, num_param in enumerate(num_param_list):
                grad = target_grad[:, offset:offset + num_param]

                # selected_bases = self.selected_bases_list[i].squeeze().T
                selected_pca = self.selected_pca_group_list[i].squeeze()
                # print(selected_bases.shape)
                selected_pca_components_tensor = selected_pca.T if Config.GEP_HISTORY_GRADS > 0 else torch.from_numpy(
                    selected_pca.components_.T)

                # embedding = torch.matmul(grad, selected_bases)
                embedding_by_pca = torch.matmul(grad, selected_pca_components_tensor)
                # bases_tensor = torch.cat([selected_bases])
                # bases_reconstruction_error = check_approx_error(bases_tensor, grad)

                pca_reconstruction_error = check_approx_error(selected_pca_components_tensor, grad)

                # reconstruction_errs.append(bases_reconstruction_error)
                pca_reconstruction_errs.append(pca_reconstruction_error)

                if logging:
                    # cur_approx = torch.matmul(torch.mean(embedding, dim=0).view(1, -1), selected_bases.T).view(-1)
                    cur_approx_pca = torch.matmul(torch.mean(embedding_by_pca, dim=0).view(1, -1),
                                                  selected_pca_components_tensor.T).view(-1)
                    cur_target = torch.mean(grad, dim=0)
                    cur_target_sqr_norm = torch.sum(torch.square(cur_target))
                    # cur_error = torch.sum(torch.square(cur_approx - cur_target)) / cur_target_sqr_norm
                    # print('group %d, param: %d, num of bases: %d, group wise approx error: %.2f%%' % (
                    #     i, num_param, self.num_bases_list[i], 100 * cur_error.item()))
                    #
                    cur_error_pca = torch.sum(torch.square(cur_approx_pca - cur_target)) / cur_target_sqr_norm
                    print('group wise approx PRIVATE error pca: %.2f%%' % (100 * cur_error_pca.item()))

                    # if i in self.approx_error:
                    #     self.approx_error[i].append(cur_error.item())
                    # else:
                    #     self.approx_error[i] = []
                    #     self.approx_error[i].append(cur_error.item())

                    if i in self.approx_error_pca:
                        self.approx_error_pca[i].append(cur_error_pca.item())
                    else:
                        self.approx_error_pca[i] = []
                        self.approx_error_pca[i].append(cur_error_pca.item())

                # embedding_list.append(embedding)
                embedding_by_pca_list.append(embedding_by_pca)
                offset += num_param
                del grad, embedding_by_pca, selected_pca
                # del grad, embedding, selected_bases, embedding_by_pca, selected_pca

            # concatenated_embedding = torch.cat(embedding_list, dim=1)
            concatenated_embedding_pca = torch.cat(embedding_by_pca_list, dim=1)

            # clipped_embedding = clip_column(concatenated_embedding, clip=self.clip0, inplace=False)
            clipped_embedding_pca = clip_column(concatenated_embedding_pca, clip=self.clip0, inplace=False)
            # if logging:
            #     norms = torch.norm(clipped_embedding, dim=1)
            #     norms_pca = torch.norm(clipped_embedding_pca, dim=1)
            #     print('power: average norm of clipped embedding: ', torch.mean(norms).item(), 'max norm: ',
            #           torch.max(norms).item(), 'median norm: ', torch.median(norms).item())
            #     print('pca: average norm of clipped embedding: ', torch.mean(norms_pca).item(), 'max norm: ',
            #           torch.max(norms_pca).item(), 'median norm: ', torch.median(norms_pca).item())
            # avg_clipped_embedding = torch.sum(clipped_embedding, dim=0) / self.batch_size
            avg_clipped_embedding_pca = torch.sum(clipped_embedding_pca, dim=0) / self.batch_size
            del clipped_embedding_pca
            # del clipped_embedding, clipped_embedding_pca

            # no_reduction_approx = self.get_approx_grad(concatenated_embedding)
            no_reduction_approx_pca = self.get_approx_grad(concatenated_embedding_pca)
            if Config.GEP_HISTORY_GRADS > 0:
                # min_shape = min(target_grad.shape[1], no_reduction_approx.shape[1])
                min_shape = min(target_grad.shape[1], no_reduction_approx_pca.shape[1])
                # residual_gradients = target_grad[:, :min_shape] - no_reduction_approx[:, :min_shape]
                residual_gradients_pca = target_grad[:, :min_shape] - no_reduction_approx_pca[:, :min_shape]
            else:
                # residual_gradients = target_grad - no_reduction_approx
                residual_gradients_pca = target_grad - no_reduction_approx_pca

            if logging:
                # norms = torch.norm(clipped_residual_gradients, dim=1)
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

            # clip_column(residual_gradients, clip=self.clip1)  # inplace clipping to save memory
            clip_column(residual_gradients_pca, clip=self.clip1)  # inplace clipping to save memory
            # clipped_residual_gradients = residual_gradients
            clipped_residual_gradients_pca = residual_gradients_pca
            # if logging:
            #     # norms = torch.norm(clipped_residual_gradients, dim=1)
            #     norms_pca = torch.norm(clipped_residual_gradients_pca, dim=1)
            #     # print('power: average norm of clipped residual gradients: ', torch.mean(norms).item(), 'max norm: ',
            #     #       torch.max(norms).item(), 'median norm: ', torch.median(norms).item())
            #     print('pca: average norm of clipped residual gradients: ', torch.mean(norms_pca).item(), 'max norm: ',
            #           torch.max(norms_pca).item(), 'median norm: ', torch.median(norms_pca).item())

            # avg_clipped_residual_gradients = torch.sum(clipped_residual_gradients, dim=0) / self.batch_size
            avg_clipped_residual_gradients_pca = torch.sum(clipped_residual_gradients_pca, dim=0) / self.batch_size
            avg_target_grad = torch.sum(target_grad, dim=0) / self.batch_size
            # del no_reduction_approx, residual_gradients, clipped_residual_gradients, no_reduction_approx_pca, \
            #     residual_gradients_pca, clipped_residual_gradients_pca

            if Config.GEP_USE_PCA:
                return avg_clipped_embedding_pca.view(-1), avg_clipped_residual_gradients_pca.view(
                    -1), avg_target_grad.view(-1)
            else:
                pass
                # return avg_clipped_embedding.view(-1), None, avg_target_grad.view(-1)
                # return avg_clipped_embedding.view(-1), avg_clipped_residual_gradients.view(-1), avg_target_grad.view(-1)
