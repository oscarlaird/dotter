# refer to math/tex/chapters/stimulus.tex
# subsection: "A Hierarchical Temporal Model of User Statistics"
import torch
import torch.nn as nn


class VNode(nn.Module):  # variational distribution node
    def __init__(self, data, priors: torch.Tensor, height=None, root=True) -> 'VNode':
        super().__init__()
        height = height if height is not None else VNode.data_height(data)
        assert len(priors) == 2**height, "priors must be of length 2**height"
        if height == 0:
            assert type(data) == float, "data must be a float"
            val = torch.empty(1, 2)
            val[0, 0] = data
            val[0, 1] = -10  # very tight deviation, essentially forcing constant
            # intentionally do not register as a parameter when leaf
            down = None
        else:
            if root:
                val = torch.empty(2**height, 2)
                val[:, 0] = priors
                val[:, 1] = -10  # very tight deviation, essentially forcing constant
                # intentionally do not register as a parameter when root
            else:
                val = torch.empty(2**height, 2)
                val[:, 0] = priors
                val[:, 1] = 0  # tune this for convergence speed; corresponds to sigma=1, a relatively uninformative prior
                val = nn.Parameter(val)
            down = nn.ModuleList()
            for subdata in data:
                child_priors = priors[::2]
                # even elements correspond to means for the next narrower scope
                # this means we initialize latents to the mean of their parents
                down.append(VNode(subdata, child_priors, height - 1, root=False))
        self.val = val
        self.down = down

    @staticmethod
    def data_height(data):
        height = 0
        while type(data) == list:
            height += 1
            data = data[0]
        return height

def total_entropy(v: VNode):
    # up to a constant, the entropy of a gaussian is log(sigma)
    # which is exactly what is in the second column of the variational distribution
    this_layer_entropy = v.val[:, 1].sum()
    if v.down is None:
        return this_layer_entropy
    total = this_layer_entropy
    for child in v.down:
        total += total_entropy(child)
    return total

def log_likelihood(v: VNode):
    if v.down is None:
        return 0
    this_ll = 0
    for child in v.down:
        for i in range(len(child.val)):
            c_mu, c_sigma = child.val[i]
            a_mu, a_sigma = v.val[i << 1]
            b_mu, b_sigma = v.val[i << 1 | 1]
            # deviations are stored in log-space
            a_sigma, b_sigma, c_sigma = torch.exp(a_sigma), torch.exp(b_sigma), torch.exp(c_sigma)
            # equation 6.4
            this_ll += (
                - b_mu
                - 0.5 * (
                    (
                        (c_mu - a_mu)**2
                        + c_sigma**2
                        + a_sigma**2
                    )
                    * torch.exp(-2 * b_mu - 2 * b_sigma**2)
                )
            )
    for child in v.down:
        this_ll += log_likelihood(child)
    return this_ll

def elbo(v: VNode):
    return log_likelihood(v) + total_entropy(v)
    
    
    
