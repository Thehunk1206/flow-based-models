
import torch

class ConditionalFlowMatcher:
    
    def __init__(self, sigma):
        self.sigma = sigma
    
    @staticmethod
    def pad_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        if isinstance(t, (float, int)):
            return t
    
        return t.reshape(-1, *([1] * (x.dim() - 1)))

    def sample_noise_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x)

    def get_mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        return (1 - t) * x0 + t * x1
    
    def get_sigma_t(self) -> torch.Tensor:
        return self.sigma
    
    def sample_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:

        sigma_t = self.get_sigma_t().type_as(x0)
        sigma_t = self.pad_t_like_x(sigma_t, x0)
        mu_t = self.get_mu_t(x0, x1, t)

        return mu_t + sigma_t * epsilon

    def get_conditional_vector_field(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        returns conditional vector field ut(x1|x0) = x1 - x0
        """
        return x1 - x0

    def sample_t(self):
        return torch.rand()
    
    def get_sample_location_and_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        t = self.sample_t()
        t = self.pad_t_like_x(t, x0)
        eps = self.sample_noise_like(x0)

        xt = self.sample_xt(x0, x1, t, eps)

        ut = self.get_conditional_vector_field(x0, x1, t)

        return xt, ut