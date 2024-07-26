import numpy as np
import torch


def sample_actions(mu, sigma):
    '''
    input:
        mu: [B, d]
        sigma: [B, 1]
    output:
        mu: [B, d]
    '''
    
    assert mu.shape[0] == sigma.shape[0]
    assert torch.is_tensor(mu) == torch.is_tensor(sigma)
    
    b = mu.shape[0]
    d = mu.shape[1]
    torch_flag = torch.is_tensor(mu)
    
    if torch_flag:
        noise = torch.randn((b, d)) * sigma
        out = mu + noise
        
    else:
        noise = np.random.normal(size=(b, d)) * sigma
        out = mu + noise
        
    return out
    

def is_valid_state(state):
    
    G = np.matmul(state, state.T)
    np.fill_diagonal(G, 0)
    return (np.logical_and(G>=-1, G<=1/2)).all()


def is_equal(state1, state2):
    return np.linalg.norm(state1 - state2) < 1e-13


def cart_to_polar(x):
    '''
    input:
        x: [n, d] or [B, n, d].
    output:
        [n, d] or [B, n, d], as $r$ is defaulted by 1.
    '''
    # CHECK
    
    d = x.shape[-1]
    n = x.shape[-2]
    if len(x.shape) == 3:
        b = x.shape[0]
        x = x.reshape(-1, d)
        n = b * n
    else:
        b = None    
    
    if torch.is_tensor(x):
        x: torch.Tensor
            
        assert torch.norm(torch.norm(x, dim=1) - 1) < 1e-5, "The cartesian coord is not 1-norm."
        polar = torch.zeros((n, d))
        tmp = 1
        
        for idx in range(d-1, 0, -1):
            polar[:, idx] = torch.arccos(x[:, idx] / (tmp + 1e-31))
            tmp = tmp * torch.sin(polar[:, idx])
        polar[:, 0] = (x[:, 0] >= 0)
        
        if b is not None:
            polar = polar.reshape((b, -1, d))
    
    else:
        x: np.ndarray
        
        assert np.linalg.norm(np.linalg.norm(x, axis=1) - 1) < 1e-5, "The cartesian coord is not 1-norm."
        polar = np.zeros((n, d))
        tmp = 1
        
        for idx in range(d-1, 0, -1):
            polar[:, idx] = np.arccos(x[:, idx] / (tmp + 1e-31))
            tmp = tmp * np.sin(polar[:, idx])
        polar[:, 0] = (x[:, 0] >= 0)
        if b is not None:
            polar = polar.reshape((b, -1, d))       
            
    return polar 


def polar_to_cart(polar):
    '''
    input:
        polar: [n, d-1] or [B, n, d-1]
    output:
        [n, d] or [B, n, d].
    '''
    
    d = polar.shape[-1]
    n = polar.shape[-2]
    if len(polar.shape) == 3:
        b = polar.shape[0]
        polar = polar.reshape(-1, d-1)
        n = b * n
    else:
        b = None  
        
    if torch.is_tensor(polar):
        polar: torch.Tensor
        
        x = torch.zeros((n, d))
        tmp = 1
        
        for idx in range(d-1, 0, -1):
            x[:, idx] = torch.cos(polar[:, idx]) * tmp
            tmp = torch.sin(polar[:, idx]) * tmp
        
        x[:, 0] = tmp * (2 * (polar[:, 0] >= .5) - 1)
        
        if b is not None:
            x = x.reshape((b, -1, d))
            
    else:
        polar: np.ndarray
        
        x = np.zeros((n, d))
        tmp = 1
        
        for idx in range(d-1, 0, -1):
            x[:, idx] = np.cos(polar[:, idx]) * tmp
            tmp = np.sin(polar[:, idx]) * tmp
        x[:, 0] = tmp * (2 * (polar[:, 0] >= .5) - 1)
        
        if b is not None:
            x = x.reshape((b, -1, d))
        
    return x


if __name__ == '__main__':
    
    cart = np.array([
        [-1., 0, 0],
        [0, -1., 0],
        [0, 0, -1.],
        [-1/np.sqrt(2), 0, 1/np.sqrt(2)]
    ])
    # polar = cart_to_polar(cart)
    # import pdb; pdb.set_trace()
    # cart = polar_to_cart(polar)
    # import pdb; pdb.set_trace()
    
    for _ in range(10000):
        polar = cart_to_polar(cart)
        cart = polar_to_cart(polar)
        
    import pdb; pdb.set_trace()
    
    # mu = torch.tensor([
    #     [1, 0, 0,],
    #     [0, 1, 0],
    #     [0, 0, 1],
    #     [0, 0, 1]
    # ])
    # sigma = torch.tensor([1, 2, 3, 4]).reshape((4, 1))
    # out = sample_actions(mu, sigma)
    # import pdb; pdb.set_trace()