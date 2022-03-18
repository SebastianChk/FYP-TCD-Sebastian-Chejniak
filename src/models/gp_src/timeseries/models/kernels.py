from gpflow.kernels import Matern32, Constant
from gpflow.utilities import set_trainable, to_default_float
from tensorflow_probability.python.distributions import LogNormal

    
class Kernel_Comb3:
    def __new__(
        cls,
        linear_variance=1.0,
        mat1_variance=1.0,
        mat1_lengthscale=0.5,
        use_priors=False,
        **kwargs
    ):
        cost_kern = Constant(linear_variance, **kwargs)
        base_k1 = Matern32(
            mat1_variance, mat1_lengthscale, **kwargs
        )

        k = cost_kern+base_k1


        return k #+ per_kern 
