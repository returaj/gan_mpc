"""runner code for expert prediction model."""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from gan_mpc import data_buffers, data_loader, data_normalizer, utils
from gan_mpc.expert import expert_model, trainer


def get_trainstate(model, params, tx):
    def predict_fn(params, batch_xseq, teacher_forcing):
        batch_carry = model.get_init_carry(batch_xseq)
        _, out = model.apply(params, batch_carry, batch_xseq, teacher_forcing)
        return out

    return train_state.TrainState.create(
        apply_fn=predict_fn, params=params, tx=tx
    )


def get_model(config, state_size, action_size):
    expert_model_config = config.expert_prediction.model
    model = expert_model.ExpertModel.get_model(
        model_config=expert_model_config, x_size=state_size, u_size=action_size
    )
    return model, expert_model_config


def get_params(config, model, state_size):
    seed = config.seed
    args = model.get_init_params(seed, 1, 1, state_size)
    return model.init(*args)


def get_optimizer(config):
    lr = config.expert_prediction.train.learning_rate
    return optax.chain(
        optax.clip_by_global_norm(max_norm=100.0), optax.adam(lr)
    )


def get_normalizer(norm_config):
    if norm_config.state == "standard_norm":
        state_normalizer = data_normalizer.StandardNormalizer()
    else:
        state_normalizer = data_normalizer.IdentityNormalizer()

    if norm_config.action == "identity":
        action_normalizer = data_normalizer.IdentityNormalizer()
    else:
        raise Exception(
            f"Please set appropriate action normalizer. Given: {norm_config.action}"
        )

    return data_normalizer.JointNormalizer(
        state_normalizer=state_normalizer, action_normalizer=action_normalizer
    )


def run(config_path=None):
    config = utils.get_config(config_path)
    key = jax.random.PRNGKey(config.seed)

    env_type, env_name = config.env.type, config.env.expert.name
    state_size, action_size = utils.get_state_action_size(env_type, env_name)

    model, model_config = get_model(config, state_size, action_size)
    params = get_params(config, model, state_size)
    tx = get_optimizer(config)
    trainstate = get_trainstate(model, params, tx)

    normalizer = get_normalizer(config.mpc.normalizer)
    dataloader = data_loader.DataLoader(
        config=config, normalizer=normalizer
    ).init()
    key, subkey = jax.random.split(key)
    dataset = dataloader.get_expert_dataset(subkey)

    train_config = config.expert_prediction.train
    trainstate, train_loss, test_loss = trainer.train(
        trainstate=trainstate,
        dataset=dataset,
        num_epochs=train_config.num_epochs,
        batch_size=train_config.batch_size,
        key=key,
        discount_factor=train_config.discount_factor,
        teacher_forcing_factor=train_config.teacher_forcing_factor,
        print_step=train_config.print_step,
    )

    env = utils.get_imitator_env(config=config)

    @jax.jit
    def policy_fn(params, histroy_x, history_u):
        del history_u
        histroy_x = jnp.expand_dims(histroy_x, axis=0)
        _, batch_useq = trainstate.apply_fn(params, histroy_x, True)
        return batch_useq[0][-1]

    buffer = data_buffers.Buffer(
        maxlen=train_config.seqlen, normalizer=dataloader.normalizer
    )
    avg_reward = utils.avg_run_dm_policy(
        env=env,
        policy_fn=policy_fn,
        params=trainstate.params,
        buffer=buffer,
        num_runs=3,
        max_interactions=1000,
    )

    save_config = {
        "env": config.env.to_dict(),
        "loss": {
            "train_loss": round(float(train_loss), 5),
            "test_loss": round(float(test_loss), 5),
        },
        "model": model_config.to_dict(),
        "train": train_config.to_dict(),
        "avg_reward": round(avg_reward, 2),
    }

    dir_path = f"trained_models/expert/{env_type}/{env_name}/"
    utils.save_all_args(dir_path, trainstate.params, save_config)


if __name__ == "__main__":
    config_path = "config/l2_hyperparameters.yaml"
    run(config_path=config_path)
