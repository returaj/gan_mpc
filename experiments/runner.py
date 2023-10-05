"""run via command line."""

from gan_mpc.config import load_config
from gan_mpc.experiments import arguments
from gan_mpc.gan import runner as gan_runner
from gan_mpc.norm import runner as norm_runner


def update_env(config, env_name, keys, values):
    config.env.expert.name = env_name
    config.env.imitator.name = env_name
    config.env.imitator.physics = []
    if keys is not None:
        config.env.imitator.physics = [
            {"key": k, "value": v} for k, v in zip(keys, values)
        ]
    return config


def update_mpc(config, args):
    mpc_config = config.mpc
    mpc_config.horizon = args.horizon
    mpc_config.history = args.history
    mpc_config.model.expert.load_id = args.expert_model
    mpc_config.train.num_epochs = args.num_epochs
    mpc_config.evaluate.num_runs_for_avg = args.num_eval
    mpc_config.evaluate.save_video = args.save_video
    return config


def main(args):
    config = load_config.Config.from_yaml(args.config)
    config = update_env(config, args.env_name, args.keys, args.values)
    config = update_mpc(config=config, args=args)

    if args.algo == "l2":
        norm_runner.run(config=config)
    elif args.algo == "gan":
        gan_runner.run(config=config)
    else:
        raise Exception(f"Given {args.algo} algo not found.")


if __name__ == "__main__":
    parser = arguments.get_arguments()
    args = parser.parse_args()
    main(args)
