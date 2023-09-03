"""runner code for norm based policy."""

import jax
import jax.numpy as jnp
import optax

from gan_mpc import utils
from gan_mpc.norm import l2_policy, trainer


def get_policy(config, x_size, u_size):
    cost, _ = utils.get_cost_model(config)
    dynamics, _ = utils.get_dynamics_model(config, x_size)
    expert = utils.get_expert_model(config, x_size, u_size)
    policy = l2_policy.L2MPC(
        config=config,
        cost_model=cost,
        dynamics_model=dynamics,
        expert_model=expert,
    )
    return policy, config.mpc


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
    labels = utils.get_masked_labels(
        all_vars=params.keys(),
        masked_vars=config.mpc.train.cost.no_grads,
        tx_key="tx",
        zero_key="zero",
    )
    lr = config.mpc.train.cost.learning_rate
    tx = optax.chain(optax.clip_by_global_norm(max_norm=100.0), optax.adam(lr))
    opt = optax.multi_transform(
        {"tx": tx, "zero": optax.set_to_zero()}, labels
    )
    opt_state = opt.init(params)
    return opt, opt_state


def get_dataset(config, dataset_path, key, train_split=0.8):
    X, Y = utils.get_policy_training_dataset(config, dataset_path)
    X, Y = jnp.array(X), jnp.array(Y)
    data_size = X.shape[0]
    split_pos = int(data_size * train_split)
    _, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, data_size)
    train_dataset = X[perm[:split_pos]], Y[perm[:split_pos]]
    test_dataset = X[perm[split_pos:]], Y[perm[split_pos:]]
    return (train_dataset, test_dataset)


def run(config_path, dataset_path=None):
    config = utils.get_config(config_path)
    key = jax.random.PRNGKey(config.seed)

    x_size, u_size = utils.get_state_action_size(
        env_type=config.env.type, env_name=config.env.expert.name
    )
    policy, policy_config = get_policy(config, x_size, u_size)
    params = get_params(policy, config, x_size, u_size)
    opt, opt_state = get_optimizer(config, params)
    dataset = get_dataset(config, dataset_path, key)

    params, train_loss, test_loss = trainer.train(
        policy_args=(policy, params),
        opt_args=(opt, opt_state),
        dataset=dataset,
        num_epochs=config.mpc.train.cost.num_epochs,
        batch_size=config.mpc.train.cost.batch_size,
        key=key,
        print_step=config.mpc.train.cost.print_step,
    )

    save_config = {
        "loss": {
            "train_loss": round(float(train_loss), 5),
            "test_loss": round(float(test_loss), 5),
        },
        "policy": policy_config.to_dict(),
    }

    env_type, env_name = config.env.type, config.env.expert.name
    dir_path = f"trained_models/imitator/{env_type}/{env_name}/"
    utils.save_params(
        params=params, config_dict=save_config, dir_path=dir_path
    )


if __name__ == "__main__":
    config_path = "config/l2_hyperparameters.yaml"
    run(config_path=config_path)
