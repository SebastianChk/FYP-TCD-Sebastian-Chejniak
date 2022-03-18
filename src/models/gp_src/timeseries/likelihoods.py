import tensorflow as tf
from gpflow.base import Parameter
from gpflow.likelihoods.base import ScalarLikelihood
from gpflow.utilities import positive
from gpflow.config import default_float



class NegativeBinomial(ScalarLikelihood):  # pylint: disable=too-many-ancestors
    r"""
    The negative binomial distribution with pmf:

    .. math::

        NB(y \mid \mu, \psi) =
            \frac{\Gamma(y + \psi)}{y! \Gamma(\psi)}
            \left( \frac{\mu}{\mu + \psi} \right)^y
            \left( \frac{\psi}{\mu + \psi} \right)^\psi

    where :math:`\mu = \exp(\nu)`. Its expected value is
    :math:`\mathbb{E}[y] = \mu ` and variance
    :math:`Var[Y] = \mu + \frac{\mu^2}{\psi}`.
    """

    def __init__(self, psi=1.0, **kwargs):
        super().__init__(**kwargs)
        self.invlink = tf.exp
        self.psi = Parameter(psi, transform=positive())

    def _scalar_log_prob(self, F, Y):
        mu = self.invlink(F)
        mu_psi = mu + self.psi
        psi_y = self.psi + Y
        f1 = (
            tf.math.lgamma(psi_y)
            - tf.math.lgamma(Y + 1.0)
            - tf.math.lgamma(self.psi)
        )
        f2 = Y * tf.math.log(mu / mu_psi)
        f3 = self.psi * tf.math.log(self.psi / mu_psi)
        return f1 + f2 + f3

    def _conditional_mean(self, F):
        return self.invlink(F)

    def _conditional_variance(self, F):
        mu = self.invlink(F)
        return mu + tf.pow(mu, 2) / self.psi
    
class NegativeBinomial_alternative(ScalarLikelihood):
    def __init__(self, alpha= 1.0,invlink=tf.exp,scale=1.0,nb_scaled=False, **kwargs):
        super().__init__( **kwargs)
        self.alpha = Parameter(alpha,
                               transform= positive(),
                               dtype=default_float())
        self.scale = Parameter(scale,trainable=False,dtype=default_float())
        self.invlink = invlink
        self.nb_scaled = nb_scaled

    def _scalar_log_prob(self, F, Y): 
        """
        P(Y) = Gamma(k + Y) / (Y! Gamma(k)) * (m / (m+k))^Y * (1 + m/k)^(-k)
        """
        '''
        m = self.invlink(F)
        k = 1 / self.alpha
                       
        return tf.lgamma(k + Y) - tf.lgamma(Y + 1) - tf.lgamma(k) + Y * tf.log(m / (m + k)) - k * tf.log(1 + m * self.alpha) 
        
        '''
        if self.nb_scaled == True:
            return negative_binomial(self.invlink(F)*self.scale , Y, self.alpha)
        else:  
            return negative_binomial(self.invlink(F) , Y, self.alpha)
    
    def _conditional_mean(self, F):
        if self.nb_scaled == True:
            return self.invlink(F)* self.scale
        else:  
            return self.invlink(F)
      
    def _conditional_variance(self, F):
        if self.nb_scaled == True:
            m = self.invlink(F) * self.scale
        else:
            m = self.invlink(F)
        return m + m**2 * self.alpha

def negative_binomial(m, Y, alpha):
        k = 1 / alpha
        return tf.math.lgamma(k + Y) - tf.math.lgamma(Y + 1) - tf.math.lgamma(k) + Y * tf.math.log(m / (m + k)) - k * tf.math.log(1 + m * alpha)

class ZeroInflatedNegativeBinomial(ScalarLikelihood):
    def __init__(self, alpha = 1.0,km = 1.0, invlink=tf.exp,  **kwargs):
        super().__init__( **kwargs)
        self.alpha = Parameter(alpha,
                               transform= positive(),
                               dtype=default_float())
        self.km = Parameter(km,
                           transform= positive(),
                           dtype=default_float())
        
        self.invlink = invlink

    def _scalar_log_prob(self, F, Y):
        m = self.invlink(F)
        psi = 1. - (m / (self.km + m))
        comparison = tf.equal(Y, 0)
        nb_zero = - tf.math.log(1. + m * self.alpha) / self.alpha
        log_p_zero = tf.reduce_logsumexp([tf.math.log(psi), tf.math.log(1.-psi) + nb_zero], axis=0)
        log_p_nonzero = tf.math.log(1.-psi) + negative_binomial(m, Y, self.alpha)
        return tf.where(comparison, log_p_zero, log_p_nonzero)

    def _conditional_mean(self, F):
        m = self.invlink(F)
        psi = 1. - (m /(self.km + m))
        return m * (1-psi) 
      
    def _conditional_variance(self, F):
        m = self.invlink(F)
        psi = 1. - (m /(self.km + m))
        return m * (1-psi)*(1 + (m * (psi+self.alpha)))
