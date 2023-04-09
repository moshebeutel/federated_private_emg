# Taken from https://github.com/dayu11/Gradient-Embedding-Perturbation

import torch
import torch.nn as nn
import numpy as np
import math

# package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

from common.utils import labels_to_consecutive, flatten_tensor


#
# def flatten_tensor(tensor_list):
#     for i in range(len(tensor_list)):
#         tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
#     flatten_param = torch.cat(tensor_list, dim=1)
#     del tensor_list
#     return flatten_param


@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


def clip_column(tsr, clip=1.0, inplace=True):
    if (inplace):
        if torch.cuda.is_available():
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
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]

    num_bases = min(num_bases, num_p)
    L = torch.normal(0, 1.0, size=(pub_grad.shape[1], num_bases), device=pub_grad.device)
    for i in range(power_iter):
        R = torch.matmul(pub_grad, L)  # n x k
        L = torch.matmul(pub_grad.T, R)  # p x k
        orthogonalize(L)
    error_rate = check_approx_error(L, pub_grad)
    return L, num_bases, error_rate


class GEP(nn.Module):

    def __init__(self, num_bases, batch_size, clip0=1, clip1=1, power_iter=1):
        super(GEP, self).__init__()

        self.num_bases = num_bases
        self.clip0 = clip0
        self.clip1 = clip1
        self.power_iter = power_iter
        self.batch_size = batch_size
        self.approx_error = {}

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
        # print('get_anchor_gradient')
        outputs = net(self.public_inputs)
        loss = loss_func(outputs.cpu(), self.public_targets.cpu())
        with backpack(BatchGrad()):
            # print('get_anchor_gradients. before loss.backward()')
            loss.backward()
        cur_batch_grad_list = []
        for p in net.parameters():
            # print('get_anchor_gradients. p.grad_batch.shape', p.grad_batch.shape)
            if hasattr(p, 'grad_batch'):
                cur_batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
                del p.grad_batch
        return flatten_tensor(cur_batch_grad_list)

    def get_anchor_space(self, net, loss_func, logging=False):
        # print('get_anchor_space')
        anchor_grads = self.get_anchor_gradients(net, loss_func)  # \
        # if self.selected_bases_list \
        # else torch.ones((self.batch_size, self.num_params))
        # print('get_anchor_space. anchor_grads.shape ', anchor_grads.shape)
        with torch.no_grad():
            num_param_list = self.num_param_list
            num_anchor_grads = anchor_grads.shape[0]
            num_group_p = len(num_param_list)

            selected_bases_list = []
            num_bases_list = []
            pub_errs = []

            sqrt_num_param_list = np.sqrt(np.array(num_param_list))
            # *** Cancel normalization to avoid all zeros when casting to int in TOY_STORY
            # num_bases_list = self.num_bases * (sqrt_num_param_list / np.sum(sqrt_num_param_list))
            num_bases_list = self.num_bases * sqrt_num_param_list
            num_bases_list = num_bases_list.astype(np.int)

            total_p = 0
            offset = 0
            device = next(net.parameters()).device
            for i, num_param in enumerate(num_param_list):
                pub_grad = anchor_grads[:, offset:offset + num_param]
                # print(f'i = {i}, num_param = {num_param}. pub_grad.shape {pub_grad.shape}')
                offset += num_param

                num_bases = num_bases_list[i]

                selected_bases, num_bases, pub_error = get_bases(pub_grad, num_bases, self.power_iter, logging)
                pub_errs.append(pub_error)

                num_bases_list[i] = num_bases
                selected_bases_list.append(selected_bases.to(device))

            self.selected_bases_list = selected_bases_list
            # print(f'selected bases list len {len(self.selected_bases_list)}')
            # print('self.selected_bases_list', self.selected_bases_list)
            self.num_bases_list = num_bases_list
            self.approx_errors = pub_errs
        del anchor_grads

    def forward(self, target_grad, logging=True):
        # print('gep forward')
        # if 'selected_bases_list' not in self.__dict__.keys():
        #     print('selected_bases_list not in dict. return')
        #     return target_grad

        with torch.no_grad():
            num_param_list = self.num_param_list
            embedding_list = []

            offset = 0
            if logging:
                print('group wise approx error')

            # target_grad = flatten_tensor(
            #     [net_param.grad_batch.reshape(net_param.grad_batch.shape[0], -1) for net_param in
            #      self.net_wrapper.net.parameters()])
            # print('target_grad.shape', target_grad.shape)

            for i, num_param in enumerate(num_param_list):
                grad = target_grad[:, offset:offset + num_param]
                # print(f'i = {i}, num_param = {num_param}. grad.shape {grad.shape}')
                # print(f'selected bases list len {len(self.selected_bases_list)}')
                selected_bases = self.selected_bases_list[i]
                # print(selected_bases.shape)
                embedding = torch.matmul(grad, selected_bases)
                num_bases = self.num_bases_list[i]
                if logging:
                    cur_approx = torch.matmul(torch.mean(embedding, dim=0).view(1, -1), selected_bases.T).view(-1)
                    cur_target = torch.mean(grad, dim=0)
                    cur_error = torch.sum(torch.square(cur_approx - cur_target)) / torch.sum(torch.square(cur_target))
                    print('group %d, param: %d, num of bases: %d, group wise approx error: %.2f%%' % (
                        i, num_param, self.num_bases_list[i], 100 * cur_error.item()))
                    if i in self.approx_error:
                        self.approx_error[i].append(cur_error.item())
                    else:
                        self.approx_error[i] = []
                        self.approx_error[i].append(cur_error.item())

                embedding_list.append(embedding)
                offset += num_param

            concatnated_embedding = torch.cat(embedding_list, dim=1)
            clipped_embedding = clip_column(concatnated_embedding, clip=self.clip0, inplace=False)
            if logging:
                norms = torch.norm(clipped_embedding, dim=1)
                print('average norm of clipped embedding: ', torch.mean(norms).item(), 'max norm: ',
                      torch.max(norms).item(), 'median norm: ', torch.median(norms).item())
            avg_clipped_embedding = torch.sum(clipped_embedding, dim=0) / self.batch_size

            no_reduction_approx = self.get_approx_grad(concatnated_embedding)
            residual_gradients = target_grad - no_reduction_approx
            clip_column(residual_gradients, clip=self.clip1)  # inplace clipping to save memory
            clipped_residual_gradients = residual_gradients
            if logging:
                norms = torch.norm(clipped_residual_gradients, dim=1)
                print('average norm of clipped residual gradients: ', torch.mean(norms).item(), 'max norm: ',
                      torch.max(norms).item(), 'median norm: ', torch.median(norms).item())

            avg_clipped_residual_gradients = torch.sum(clipped_residual_gradients, dim=0) / self.batch_size
            avg_target_grad = torch.sum(target_grad, dim=0) / self.batch_size
            return avg_clipped_embedding.view(-1), avg_clipped_residual_gradients.view(-1), avg_target_grad.view(-1)
