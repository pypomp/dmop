"""Two-pass filters from cholera_mop.ipynb, cell 2 (zero-based).

The fixed PRNGKey(0) at every month is retained deliberately.
"""
from functools import partial
import jax
import jax.numpy as np
from jax import jit
from legacy_model import rinit, rprocess, dmeasure

def normalize_weights(weights):
    mw = np.max(weights)
    loglik_t = mw + np.log(np.nansum(np.exp(weights - mw))) # p(y_t | x_{t,1:J}, \theta)
    norm_weights = weights - loglik_t
    return norm_weights, loglik_t

def resample_key(norm_weights, key):
    J = norm_weights.shape[-1]

    unifs = (jax.random.uniform(key=key)+np.arange(J)) / J

    csum = np.cumsum(np.exp(norm_weights))
    counts = np.repeat(np.arange(J),
                       np.histogram(unifs,
                        bins=np.pad(csum/csum[-1], pad_width=(1,0)),
                            density=False)[0].astype(int),
                      total_repeat_length=J)

    #if len(counts)<J:
    #    counts = np.hstack([counts, np.zeros(J-len(counts))]).astype(int)
    return counts

def resampler_key(counts, particlesP, norm_weights, key):
    J = norm_weights.shape[-1]
    counts = resample_key(norm_weights, key)
    particlesF = particlesP[counts]
    norm_weights = norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - np.log(J)
    return counts, particlesF, norm_weights


def pfilter_debug_helper(t, inputs):
    particles, meas_particles, particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)
    key = jax.random.PRNGKey(0)

    key, *keys = jax.random.split(key, num=J+1)
    keys = np.array(keys)

    # Get prediction particles
    particlesP = rprocess(particlesF, theta, keys, covars)# if t>0 else particlesF

    measurements = dmeasure(ys[t], particlesP, theta)

    # Multiply weights by measurement model result
    weights = norm_weights + measurements

    # Obtain normalized weights
    norm_weights, loglik_t = normalize_weights(weights)
    loglik += loglik_t

    meas_particles = meas_particles.at[t].set(measurements)

    counts, particlesF, norm_weights = resampler_key(counts, particlesP, norm_weights, key)

    #jax.debug.print(loglik, loglik_t)
    particles = particles.at[t+1].set(particlesF)
    return [particles, meas_particles, particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key]


def pfilter_debug(theta, ys, J, covars=None, thresh=-1, key=jax.random.PRNGKey(0)):

    particlesF = rinit(theta, J, covars=covars)
    particles = np.zeros((len(ys)+1, J, particlesF.shape[-1]))
    particles = particles.at[0].set(particlesF)
    meas_particles = np.zeros((len(ys), J))
    weights = np.log(np.ones(J)/J)
    norm_weights = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    loglik = 0

    particles, meas_particles, particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=pfilter_debug_helper,
                 init_val=[particles, meas_particles, particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key])
    return -loglik, particles, meas_particles



def mop_debug_helper(t, inputs):
    particles, meas_phi, particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key = inputs
    J = len(particlesF)
    key = jax.random.PRNGKey(0)

    key, *keys = jax.random.split(key, num=J+1)
    keys = np.array(keys)

    # Discount weights by alpha in logspace
    weightsP = alpha*weightsF

    # Get prediction particles
    particlesP = rprocess(particlesF, theta, keys, covars)# if t>0 else particlesF

    measurements = dmeasure(ys[t], particlesP, theta)

    # Using before-resampling conditional likelihood
    loglik += (jax.scipy.special.logsumexp(weightsP + measurements)
               - jax.scipy.special.logsumexp(weightsP))

    # Obtain normalized measurement likelihoods for resampling
    norm_weights, loglik_phi_t = normalize_weights(jax.lax.stop_gradient(meas_phi[t]))

    # Systematic resampling according to normalized measurement likelihoods
    counts, particlesF, norm_weightsF = resampler_key(counts, particlesP, norm_weights, key)

    # Correct for theta/phi and resample
    weightsF = (weightsP + measurements - jax.lax.stop_gradient(meas_phi[t]))[counts]

    particles = particles.at[t+1].set(particlesF)
    #jax.debug.print(loglik, loglik_t)
    return [particles, meas_phi, particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key]


# test on linear gaussian toy model again
@partial(jit, static_argnums=2)
def mop_debug(theta, ys, J, meas_phi, covars=None, alpha=0.97, key=jax.random.PRNGKey(0)):

    particlesF = rinit(theta, J, covars=covars)

    particles = np.zeros((len(ys)+1, J, particlesF.shape[-1]))
    particles = particles.at[0].set(particlesF)

    weights = np.log(np.ones(J)/J)
    weightsF = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    loglik = 0

    particles, meas_phi, particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=mop_debug_helper,
                 init_val=[particles, meas_phi, particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key])

    return -loglik, particles
