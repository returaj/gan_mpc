"""NN model for expert."""

import flax.linen as nn

from gan_mpc import base


class MLPCell(nn.Module):
    num_layers: int
    num_hidden_units: int
    fout: int

    @nn.compact
    def __call__(self, carry, _):
        x = carry
        for _ in range(self.num_layers):
            x = nn.relu(nn.Dense(self.num_hidden_units)(x))
        out = nn.Dense(self.fout)(x)
        return out, out


class ScanMLP(nn.Module):
    num_layers: int
    num_hidden_units: int
    fout: int
    horizon: int

    def initialize_carry(self, *args):
        del args
        return None

    @nn.compact
    def __call__(self, carry, x):
        del carry
        mlp = nn.scan(
            MLPCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            length=self.horizon,
        )(self.num_layers, self.num_hidden_units, self.fout)

        _, out = mlp(x, None)
        return out


# class StateAction(nn.Module, base.BaseNN):
#     state_model: ScanMLP
#     action_model: ScanMLP

#     def get_init_params(self, seed, batch_size, x_size):


#     @nn.compact
#     def __call__(self, carry, x):


# class StateActionNN(nn.Module, base.BaseNN):
#     num_layers: int
#     num_hidden_units: int
#     action_model: MLP
#     state_model: MLP

#     def get_init_params(self, seed, x_size):
#         key = jax.random.PRNGKey(seed)
#         dummy_x = jnp.zeros(x_size)
#         return (key, dummy_x)

#     @nn.compact
#     def __call__(self, x):
#         q = x
#         for _ in range(self.num_layers):
#             q = nn.relu(nn.Dense(self.num_hidden_units)(q))
#         u = nn.tanh(self.action_model(q))  # u \in (-1, 1)
#         next_x = x + self.state_model(q)
#         return (u, next_x)
