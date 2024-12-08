import torch
import numpy as np

from .optimal_transport import OTPlanSampler, wasserstein
from .cond_flow_matcher import ConditionalFlowMatcher

class OptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    A Conditional Flow Matcher that uses Optimal Transport for more sophisticated 
    sampling and vector field computation.
    
    This class enhances the base Conditional Flow Matcher by:
    1. Using Optimal Transport to guide sampling between distributions
    2. Providing more flexible sampling strategies
    3. Incorporating Wasserstein distance computations
    """
    
    def __init__(self, 
                 sigma: float = 0.1, 
                 ot_method: str = 'sinkhorn', 
                 ot_reg: float = 0.05, 
                 ot_normalize_cost: bool = False):
        """
        Initialize the Optimal Transport Conditional Flow Matcher
        
        Parameters
        ----------
        sigma : float, optional
            Noise scale parameter, by default 0.1
        ot_method : str, optional
            Optimal Transport method to use, by default 'sinkhorn'
        ot_reg : float, optional
            Regularization parameter for OT solver, by default 0.05
        ot_normalize_cost : bool, optional
            Whether to normalize cost matrix in OT solver, by default False
        """
        super().__init__(sigma)
        
        # Initialize Optimal Transport Plan Sampler
        self.ot_sampler = OTPlanSampler(
            method=ot_method, 
            reg=ot_reg, 
            normalize_cost=ot_normalize_cost
        )

    def get_sample_location_and_conditional_flow(self, 
                                                 x0: torch.Tensor, 
                                                 x1: torch.Tensor, 
                                                 t: torch.Tensor = None) -> tuple:
        """
        Sample location and conditional flow using Optimal Transport
        
        Parameters
        ----------
        x0 : torch.Tensor
            Source distribution
        x1 : torch.Tensor
            Target distribution
        t : torch.Tensor, optional
            Time step, if None, sampled randomly
        
        Returns
        -------
        tuple
            Sampled location and conditional flow
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().get_sample_location_and_conditional_flow(x0, x1, t)