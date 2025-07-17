"""
src.utils.welford.py
Welford calculator for online calculation of mean and standard deviations of an image set
BoMeyering 2025
"""
import torch
import sys
from typing import Tuple

class WelfordCalculator:
    def __init__(self, device: str):
        """ Initialize the Welford calculator """
        self.M = torch.zeros(3, dtype=torch.float64).to(device) # Current channel mean vector
        self.M_old = torch.zeros(3, dtype=torch.float64).to(device) # Previous iteration channel mean vector
        self.S = torch.zeros(3, dtype=torch.float64).to(device) # S term used to calculate channel standard deviation
        self.N = torch.zeros(1, dtype=torch.long).to(device) # The total number of pixels used in the calculation

    def update(self, array: torch.Tensor) -> None:
        """
        Make an update to the state given the image tensor 'array'

        Args:
            array (torch.Tensor): a three channel array representing an image of shape (3, H, W)

        Raises:
            ValueError: If the shape is not as expected
        """
        if (len(array.shape) > 3) or (array.shape[0] != 3):
            raise ValueError(f"Passed tensor should be a 3 channel image of shape (3, H, W). Current 'array' argument is of shape {array.shape}")
        
        # Update count based on number of pixels
        num_px = array.shape[1] * array.shape[2] # Assumes C, H, W shape
        self.N += num_px

        # Calculate array mean and update running mean
        self.M_old = self.M.clone()
        m_new = array.mean(dim=(1, 2)) # Calculate mean over all pixels for each channel
        self.M += (m_new - self.M) * num_px / self.N # Update self.M

        # Calculate new S incremental term (x-M_t) * (x-M_{t-1})
        S_incr = (array - self.M.unsqueeze(-1).unsqueeze(-1)) * (array - self.M_old.unsqueeze(-1).unsqueeze(-1))
        self.S += S_incr.sum(dim=(1, 2)) # Update self.S with the incremental term

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the final channel means and standard deviations from the current class states

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A length 3 channel mean vector, and a length 3 channel standard deviation vector
        """
        self.std = torch.sqrt(self.S / self.N)

        return self.M, self.std