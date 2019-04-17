"""
Prior information for a normal-inverse-Wishart distribution
Author: Jun Lu
Contact: jun.lu.locky@gmail.com
Date: 2017
"""

class NIW(object):
    """A normal-inverse-Wishart distribution."""
    def __init__(self, m_0, k_0, v_0, S_0):
        """

        :param m_0: Prior mean for the mean vector of multivariate Gaussian distribution
        :param k_0: How strongly we believe the above prior.
        :param v_0: How strongly we believe the following prior.
        :param S_0: Proportional to prior mean for Covariance matrix of multivariate Gaussian distribution
        """
        self.m_0 = m_0
        self.k_0 = k_0
        D = len(m_0)
        assert v_0 >= D, "v_0 must be larger or equal to dimension of data"
        self.v_0 = v_0
        self.S_0 = S_0
