"""runner code for norm based policy."""

import collections

import jax
import jax.numpy as jnp
import optax

from gan_mpc import utils
from gan_mpc.norm import cost_trainer, dynamics_trainer, l2_policy
from gan_mpc.policy import eval


def get_policy(config, x_size, u_size):
    cost, _ = utils.get_cost_model(config)
    dynamics, _ = utils.get_dynamics_model(config, x_size)
    expert = utils.get_expert_model(config, x_size, u_size)
    train_policy = l2_policy.L2MPC(
        config=config,
        cost_model=cost,
        dynamics_model=dynamics,
        expert_model=expert,
    )
    eval_policy = eval.EvalMPC(
        config=config,
        cost_model=cost,
        dynamics_model=dynamics,
        expert_model=expert,
    )
    return train_policy, eval_policy, config.mpc


def get_params(policy, config, x_size, u_size):
    seed = config.seed

    carry = policy.get_dynamics_carry(jnp.zeros((1, x_size)))
    carry_size = carry.shape[-1]
    xc_size = x_size + carry_size

    mpc_weights = tuple(config.mpc.model.cost.weights.to_dict().values())
    cost_args = (seed, xc_size)
    dynamics_args = (seed, u_size)
    expert_args = (True,)
    return policy.init(mpc_weights, cost_args, dynamics_args, expert_args)


def get_optimizer(params, masked_vars, lr):
    labels = utils.get_masked_labels(
        all_vars=params.keys(),
        masked_vars=masked_vars,
        tx_key="tx",
        zero_key="zero",
    )
    tx = optax.chain(optax.clip_by_global_norm(max_norm=100.0), optax.adam(lr))
    opt = optax.multi_transform(
        {"tx": tx, "zero": optax.set_to_zero()}, labels
    )
    opt_state = opt.init(params)
    return opt, opt_state


def train(
    config,
    env,
    policy_args,
    cost_opt_args,
    dynamics_opt_args,
    buffers,
    cost_dataset,
    dynamics_dataset,
    key,
):
    train_policy, eval_policy, params = policy_args
    cost_opt, cost_opt_state = cost_opt_args
    dynamics_opt, dynamics_opt_state = dynamics_opt_args
    num_epochs = config.mpc.train.num_epochs
    print_after_n_epochs = config.mpc.train.print_after_n_epochs
    cost_config = config.mpc.train.cost
    dynamics_config = config.mpc.train.dynamics
    cost_train_losses, cost_test_losses = [], []
    # default values
    dynamics_train_losses, dynamics_test_losses = [0.0], [0.0]
    dynamics_env_rewards = [[0.0]]  # default values
    for ep in range(1, num_epochs + 1):
        key, subkey1, subkey2 = jax.random.split(key, 3)

        (
            params,
            dynamics_opt_state,
            buffers,
            epoch_dynamics_env_rewards,
            epoch_dynamics_train_losses,
            epoch_dynamics_test_losses,
            dynamics_exe_time,
        ) = dynamics_trainer.train(
            env=env,
            train_args=(train_policy, eval_policy, dynamics_opt),
            opt_state=dynamics_opt_state,
            params=params,
            dataset=dynamics_dataset,
            buffers=buffers,
            num_episodes=dynamics_config.num_episodes,
            max_interactions_per_episode=dynamics_config.max_interactions_per_episode,
            num_updates=dynamics_config.num_updates,
            batch_size=dynamics_config.batch_size,
            discount_factor=dynamics_config.discount_factor,
            teacher_forcing_factor=dynamics_config.teacher_forcing_factor,
            key=subkey1,
            id=ep,
        )

        (
            params,
            cost_opt_state,
            epoch_cost_train_losses,
            epoch_cost_test_losses,
            cost_exe_time,
        ) = cost_trainer.train(
            train_args=(train_policy, cost_opt),
            opt_state=cost_opt_state,
            params=params,
            dataset=cost_dataset,
            num_updates=cost_config.num_updates,
            batch_size=cost_config.batch_size,
            polyak_factor=cost_config.polyak_factor,
            key=subkey2,
            id=ep,
        )

        dynamics_env_rewards.extend(epoch_dynamics_env_rewards)
        dynamics_train_losses.extend(epoch_dynamics_train_losses)
        dynamics_test_losses.extend(epoch_dynamics_test_losses)

        cost_train_losses.extend(epoch_cost_train_losses)
        cost_test_losses.extend(epoch_cost_test_losses)

        if (ep % print_after_n_epochs) == 0:
            print("-----------------------------")
            print(
                f"epoch: {ep} env_reward: {sum(dynamics_env_rewards[-1]):.2f}"
            )
            print(
                f"dyna_exe_time: {dynamics_exe_time:.2f} mins, "
                f"dyna_train_loss: {dynamics_train_losses[-1]:.5f}, "
                f"dyna_test_loss: {dynamics_test_losses[-1]:.5f}"
            )
            print(
                f"cost_exe_time: {cost_exe_time:.2f} mins, "
                f"cost_train_loss: {cost_train_losses[-1]:.5f}, "
                f"cost_test_loss: {cost_test_losses[-1]:.5f}"
            )

    return (
        params,
        (dynamics_env_rewards, dynamics_train_losses, dynamics_test_losses),
        (cost_train_losses, cost_test_losses),
    )


def run(config_path, dataset_path=None):
    config = utils.get_config(config_path)
    key = jax.random.PRNGKey(config.seed)

    x_size, u_size = utils.get_state_action_size(
        env_type=config.env.type, env_name=config.env.expert.name
    )
    train_policy, eval_policy, policy_config = get_policy(
        config, x_size, u_size
    )
    params = get_params(train_policy, config, x_size, u_size)

    cost_opt_args = get_optimizer(
        params=params,
        masked_vars=config.mpc.train.cost.no_grads,
        lr=config.mpc.train.cost.learning_rate,
    )
    dynamics_opt_args = get_optimizer(
        params=params,
        masked_vars=config.mpc.train.dynamics.no_grads,
        lr=config.mpc.train.dynamics.learning_rate,
    )

    cost_dataset = cost_trainer.get_dataset(config, dataset_path, key)
    dynamics_dataset = dynamics_trainer.get_dataset(config, dataset_path)

    env = utils.get_imitator_env(
        env_type=config.env.type,
        env_name=config.env.imitator.name,
        seed=config.seed,
    )

    replay_buffer = dynamics_trainer.ReplayBuffer(
        horizon=config.mpc.horizon,
        maxlen=config.mpc.train.dynamics.replay_buffer_size,
    )
    buffer_x = collections.deque(maxlen=config.mpc.history + 1)
    buffer_u = collections.deque(maxlen=config.mpc.history)

    params, dynamics_out_args, cost_out_args = train(
        config=config,
        env=env,
        policy_args=(train_policy, eval_policy, params),
        cost_opt_args=cost_opt_args,
        dynamics_opt_args=dynamics_opt_args,
        buffers=(replay_buffer, buffer_x, buffer_u),
        cost_dataset=cost_dataset,
        dynamics_dataset=dynamics_dataset,
        key=key,
    )

    (
        dynamics_env_rewards,
        dynamics_train_losses,
        dynamics_test_losses,
    ) = dynamics_out_args

    (cost_train_losses, cost_test_losses) = cost_out_args

    avg_reward = utils.avg_run_dm_policy(
        env=env,
        policy_fn=eval_policy.get_optimal_action,
        params=params,
        buffer_x=buffer_x,
        buffer_u=buffer_u,
        max_interactions=config.mpc.evaluate.max_interactions,
        num_runs=config.mpc.evaluate.num_runs_for_avg,
    )

    save_config = {
        "loss": {
            "dynamics": {
                "train_loss": round(dynamics_train_losses[-1], 5),
                "test_loss": round(dynamics_test_losses[-1], 5),
            },
            "cost": {
                "train_loss": round(cost_train_losses[-1], 5),
                "test_loss": round(cost_test_losses[-1], 5),
            },
        },
        "reward": round(avg_reward, 2),
        "policy": policy_config.to_dict(),
    }

    env_type, env_name = config.env.type, config.env.expert.name
    dir_path = f"trained_models/imitator/{env_type}/{env_name}/l2/"

    abs_dir_path = utils.save_all_args(
        dir_path,
        params,
        save_config,
        (dynamics_env_rewards, "dynamics_env_rewards.json"),
        (dynamics_train_losses, "dynamics_train_losses.json"),
        (dynamics_test_losses, "dynamics_test_losses.json"),
        (cost_train_losses, "cost_train_losses.json"),
        (cost_test_losses, "cost_test_losses.json"),
    )

    if config.mpc.evaluate.save_video:
        utils.save_video(
            env=env,
            policy_fn=eval_policy.get_optimal_action,
            params=params,
            buffer_x=buffer_x,
            buffer_u=buffer_u,
            dir_path=abs_dir_path,
            file_path="video.mp4",
        )


if __name__ == "__main__":
    config_path = "config/l2_hyperparameters.yaml"
    run(config_path=config_path)
