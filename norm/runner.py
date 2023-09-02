"""runner code for norm based policy."""

import jax
import jax.numpy as jnp
import optax

from gan_mpc import utils
from gan_mpc.config import load_config
from gan_mpc.cost import cost_model
from gan_mpc.cost import nn as cost_nn
from gan_mpc.dynamics import dynamics_model
from gan_mpc.dynamics import nn as dynamics_nn
from gan_mpc.expert import expert_model
from gan_mpc.norm import l2_policy, trainer


def get_cost_model(config):
    model_config = config.mpc.model.cost
    mlp_config = model_config.mlp
    nn_model = cost_nn.MLP(
        num_layers=mlp_config.num_layers,
        num_hidden_units=mlp_config.num_hidden_units,
        fout=mlp_config.fout,
    )
    return cost_model.MujocoBasedModel(config, nn_model), model_config


def get_dynamics_model(config, x_size):
    model_config = config.mpc.model.dynamics
    mlp_config = model_config.mlp
    nn_model = dynamics_nn.MLP(
        num_layers=mlp_config.num_layers,
        num_hidden_units=mlp_config.num_hidden_units,
        x_out=x_size,
    )
    return dynamics_model.DynamicsModel(config, nn_model), model_config


def get_expert_model(config, x_size, u_size):
    env_type, env_name = config.env.type, config.env.expert.name
    env_id = config.expert_prediction.load.id
    saved_config_path = (
        f"trained_models/expert/{env_type}/{env_name}/{env_id}/config.json"
    )
    saved_config = utils.load_json(saved_config_path)
    model_config = load_config.Config.from_dict(saved_config["model"])
    nn_model = expert_model.ExpertModel.get_model(
        model_config=model_config, x_size=x_size, u_size=u_size
    )
    return expert_model.ExpertModel(config, nn_model)


def get_policy(config, x_size, u_size):
    cost, cost_config = get_cost_model(config)
    dynamics, dynamics_config = get_dynamics_model(config, x_size)
    expert = get_expert_model(config, x_size, u_size)
    policy = l2_policy.L2MPC(
        config=config,
        cost_model=cost,
        dynamics_model=dynamics,
        expert_model=expert,
    )
    return policy, cost_config, dynamics_config


def get_params(policy, config, x_size, u_size):
    seed = config.seed

    carry = policy.get_carry(jnp.zeros(x_size))
    carry_size = carry.shape[-1]
    xc_size = x_size + carry_size

    mpc_weights = tuple(config.mpc.model.cost.weights.to_dict().values())
    cost_args = (
        seed,
        xc_size,
    )
    dynamics_args = (
        seed,
        u_size,
    )
    expert_args = (True,)
    return policy.init(mpc_weights, cost_args, dynamics_args, expert_args)


def get_optimizer(config, params):
    lr = config.train.learning_rate
    opt = optax.chain(
        optax.clip_by_global_norm(max_norm=100.0), optax.adam(lr)
    )
    opt_state = opt.init(params)
    return opt, opt_state


def get_dataset(config, dataset_path, key, train_split=0.8):
    X, Y = utils.get_policy_training_dataset(config, dataset_path)
    data_size = X.shape[0]
    split_pos = int(data_size * train_split)
    _, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, data_size)
    train_dataset = X[perm[:split_pos]], Y[perm[:split_pos]]
    test_dataset = X[perm[split_pos:]], Y[perm[split_pos:]]
    return (train_dataset, test_dataset)


def run(config_path=None, dataset_path=None):
    config = utils.get_config(config_path)
    key = jax.random.PRNGKey(config.seed)

    x_size, u_size = utils.get_state_action_size(
        env_type=config.env.type, env_name=config.env.expert.name
    )
    policy, cost_config, dynamics_config = get_policy(config, x_size, u_size)
    params = get_params(policy, config, x_size, u_size)
    opt, opt_state = get_optimizer(config, params)
    dataset = get_dataset(config, dataset_path, key)

    params, train_loss, test_loss = trainer.train(
        policy_args=(policy, params),
        opt_args=(opt, opt_state),
        dataset=dataset,
        num_epochs=config.train.num_epochs,
        batch_size=config.train.batch_size,
        key=key,
        print_step=config.train.print_step,
    )

    save_config = {
        "loss": {
            "train_loss": round(float(train_loss), 5),
            "test_loss": round(float(test_loss), 5),
        },
        "cost": cost_config.to_dict(),
        "dynamics": dynamics_config.to_dict(),
    }

    env_type, env_name = config.env.type, config.env.expert.name
    dir_path = f"trained_models/imitator/{env_type}/{env_name}/"
    utils.save_params(
        params=params, config_dict=save_config, dir_path=dir_path
    )


if __name__ == "__main__":
    run()
