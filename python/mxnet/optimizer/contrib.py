# coding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=too-many-lines
"""Contrib optimizers."""
from ..ndarray import (NDArray, clip, contrib, mean, sqrt, square, zeros)
from .optimizer import Optimizer
import math

# convenience wrapper for Optimizer.Register
register = Optimizer.register  # pylint: disable=invalid-name

__all__ = ['GroupAdaGrad', 'Radam']


@register
class GroupAdaGrad(Optimizer):
    """Adagrad optimizer with row-wise learning rates.

    This class implements the AdaGrad optimizer described in *Adaptive
    Subgradient Methods for Online Learning and Stochastic Optimization*, and
    available at http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf but
    uses only a single learning rate for every row of the parameter array.

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad, clip_gradient)
        history += mean(square(grad), axis=1, keepdims=True)
        div = grad / sqrt(history + float_stable_eps)
        weight -= div * lr

    Weights are updated lazily if the gradient is sparse.

    For details of the update algorithm see
    :class:`~mxnet.ndarray.contrib.group_adagrad_update`.

    This optimizer accepts the following parameters in addition to those
    accepted by :class:`.Optimizer`. Weight decay is not supported.

    Parameters
    ----------
    eps: float, optional
        Initial value of the history accumulator. Avoids division by 0.

    """

    def __init__(self, eps=1e-5, **kwargs):
        super(GroupAdaGrad, self).__init__(**kwargs)
        self.float_stable_eps = eps

    def create_state(self, index, weight):
        assert len(weight.shape) == 2
        history = zeros(
            (weight.shape[0], 1), weight.context, stype=weight.stype)
        return history

    def update(self, index, weight, grad, state):
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        assert wd == 0, 'Weight decay is not supported for GroupAdaGrad'

        is_sparse = grad.stype == 'row_sparse'
        if is_sparse:
            kwargs = {
                'epsilon': self.float_stable_eps,
                'rescale_grad': self.rescale_grad
            }
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient
            contrib.group_adagrad_update(
                weight,
                grad,
                state,
                out=weight,
                lr=lr,
                **kwargs)
        else:
            grad = grad * self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
            state[:] += mean(square(grad), axis=1, keepdims=True)
            div = lr * grad / sqrt(state + self.float_stable_eps)
            weight[:] -= div


@register
class Radam(Optimizer):
    """The RAdam optimizer.

    A new variant of Adam, by introducing a term to rectify the variance
    of the adaptive learning rate.    
    
    Paper: "On the Variance of the Adaptive Learning Rate and Beyond", Liu et al. 2019,
    link: https://arxiv.org/abs/1908.03265

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional
        Small value to avoid division by 0.
    N_sma_threshhold : float, optional
        Adjustable threshold for adaptive Adam
    """
    def __init__(self, learning_rate=0.001, beta1=0.95, beta2=0.999, epsilon=1e-5,
                 N_sma_threshhold=5, **kwargs):
        super(Radam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.N_sma_threshhold = N_sma_threshhold
        self.radam_buffer = [[None,None,None] for ind in range(10)]

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]

        # preprocess grad
        grad = grad * self.rescale_grad + wd * weight
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        
        # update m_t and v_t
        m_t, v_t = state
        m_t[:] = self.beta1 * m_t + (1. - self.beta1) * grad
        v_t[:] = self.beta2 * v_t + (1. - self.beta2) * grad * grad

        #
        buffered = self.radam_buffer[int(t % 10)]
        if t == buffered[0]:
            N_sma, step_size = buffered[1], buffered[2] 
        else:
            buffered[0] = t
            beta2_t = pow(self.beta2, t)
            N_sma_max = 2. / (1. - self.beta2) - 1.
            N_sma = N_sma_max - 2. * t * beta2_t / (1. - beta2_t)
            buffered[1] = N_sma
            if N_sma > self.N_sma_threshhold:
                step_size = math.sqrt((1. - beta2_t) * (N_sma - 4.) / (N_sma_max - 4.) * (N_sma - 2.) / N_sma * N_sma_max / (N_sma_max - 2.)) / (1. - pow(self.beta1, t))
            else:
                step_size = 1. / (1. - pow(self.beta1, t))
            buffered[2] = step_size

        if N_sma > self.N_sma_threshhold:
            denom = sqrt(v_t) + self.epsilon            
            weight[:] -= (step_size * lr) * m_t / denom
        else:
            weight[:] -= (step_size * lr) * m_t