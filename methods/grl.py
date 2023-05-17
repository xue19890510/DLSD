"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional, Any, Tuple
import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torch
import torch.nn.functional as F


def shift_log(x, offset=1e-6):
    """
    First shift, then calculate log for numerical stability.
    """

    return torch.log(torch.clamp(x + offset, max=1.))


class WorstCaseEstimationLoss(nn.Module):
    r"""
    Worst-case Estimation loss from `Debiased Self-Training for Semi-Supervised Learning <https://arxiv.org/abs/2202.07136>`_
    that forces the worst possible head :math:`h_{\text{worst}}` to predict correctly on all labeled samples
    :math:`\mathcal{L}` while making as many mistakes as possible on unlabeled data :math:`\mathcal{U}`. In the
    classification task, it is defined as:

    .. math::
        loss(\mathcal{L}, \mathcal{U}) =
        \eta' \mathbb{E}_{y^l, y_{adv}^l \sim\hat{\mathcal{L}}} -\log\left(\frac{\exp(y_{adv}^l[h_{y^l}])}{\sum_j \exp(y_{adv}^l[j])}\right) +
        \mathbb{E}_{y^u, y_{adv}^u \sim\hat{\mathcal{U}}} -\log\left(1-\frac{\exp(y_{adv}^u[h_{y^u}])}{\sum_j \exp(y_{adv}^u[j])}\right),

    where :math:`y^l` and :math:`y^u` are logits output by the main head :math:`h` on labeled data and unlabeled data,
    respectively. :math:`y_{adv}^l` and :math:`y_{adv}^u` are logits output by the worst-case estimation
    head :math:`h_{\text{worst}}`. :math:`h_y` refers to the predicted label when the logits output is :math:`y`.

    Args:
        eta_prime (float): the trade-off hyper parameter :math:`\eta'`.

    Inputs:
        - y_l: logits output :math:`y^l` by the main head on labeled data
        - y_l_adv: logits output :math:`y^l_{adv}` by the worst-case estimation head on labeled data
        - y_u: logits output :math:`y^u` by the main head on unlabeled data
        - y_u_adv: logits output :math:`y^u_{adv}` by the worst-case estimation head on unlabeled data

    Shape:
        - Inputs: :math:`(minibatch, C)` where C denotes the number of classes.
        - Output: scalar.

    """

    # def __init__(self, eta_prime):
    def __init__(self):
        super(WorstCaseEstimationLoss, self).__init__()
        # self.eta_prime = eta_prime

    # def forward(self, y_l, y_l_adv, y_u, y_u_adv):
    #     _, prediction_l = y_l.max(dim=1)
    #     loss_l = self.eta_prime * F.cross_entropy(y_l_adv, prediction_l)

    #     _, prediction_u = y_u.max(dim=1)
    #     loss_u = F.nll_loss(shift_log(1. - F.softmax(y_u_adv, dim=1)), prediction_u)

    #     return loss_l + loss_u
    
    def forward(self, y_u, y_u_adv):

        _, prediction_u = y_u.max(dim=1)
        loss_u = F.nll_loss(shift_log(1. - F.softmax(y_u_adv, dim=1)), prediction_u)

        return  loss_u


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 220., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        # coeff=0.1
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1
