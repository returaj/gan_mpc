"""runner code for expert prediction model."""

import optax
from flax.training import train_state

from gan_mpc import utils
from gan_mpc.expert import nn as expert_nn
from gan_mpc.expert import trainer


def get_train_dataset(config, dataset_path=None):
    # TODO(returaj) implement this
    trajectories = utils.get_expert_trajectories(config, dataset_path)
    raise NotImplementedError


def get_trainstate(model, params, tx):
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


def get_model(config, state_size, action_size):
    expert_model_config = config.expert_prediction.model
    model_config = expert_model_config.mlp
    if not model_config.use:
        model_config = expert_model_config.lstm
    model = expert_nn.ScanLSTM(
        lstm_features=model_config.lstm_features,
        num_layers=model_config.num_layers,
        num_hidden_units=model_config.num_hidden_units,
        x_out=state_size,
        u_out=action_size,
    )
    return expert_nn.StateAction(model), expert_model_config


def get_params(config, model, state_size):
    seed = config.seed
    batch_size = config.expert_prediction.train.batch_size
    seqlen = config.expert_prediction.train.seqlen
    args = model.get_init_params(seed, batch_size, seqlen, state_size)
    return model.init(*args)


def get_optimizer(config):
    lr = config.expert_prediction.train.learning_rate
    return optax.chain(
        optax.clip_by_global_norm(max_norm=100.0), optax.adam(lr)
    )


def run(config_path=None):
    config = utils.get_config(config_path)
    env_type, env_name = config.env.type, config.env.expert.name
    state_size, action_size = utils.get_state_action_size(env_type, env_name)

    model, model_config = get_model(config, state_size, action_size)
    params = get_params(config, model, state_size)
    tx = get_optimizer(config)
    trainstate = get_trainstate(model, params, tx)

    dataset = get_train_dataset(config)

    train_config = config.expert_prediction.train
    trainstate, train_loss, test_loss = trainer.train(
        trainstate=trainstate,
        dataset=dataset,
        num_epochs=train_config.num_epochs,
        batch_size=train_config.batch_size,
        discount_factor=train_config.discount_factor,
        teacher_forcing_factor=train_config.teacher_forcing_factor,
        print_step=train_config.print_step,
    )

    loss = {train_loss: round(train_loss, 5), test_loss: round(test_loss, 5)}

    dir_path = f"trained_models/expert/{env_type}/{env_name}/model"
    utils.save_model(trainstate, model_config, train_config, loss, dir_path)


if __name__ == "__main__":
    run()
