"""Choose how to differentiate the unchanged scalar MOP proposal objective."""
import jax


def select_value_force(value_force, mode='reverse'):
    if mode == 'reverse':
        return value_force
    if mode != 'forward':
        raise ValueError(f'Unknown force differentiation mode: {mode}')

    @jax.jit
    def forward(z, key):
        def with_aux(x):
            value = value_force(x, key)[0]
            return value, value
        force, value = jax.jacfwd(with_aux, has_aux=True)(z)
        return value, force.astype(z.dtype)

    return forward
