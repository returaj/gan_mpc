"""add arguments."""

import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path")
    parser.add_argument("--algo", type=str, help="choose which algo l2 or gan")
    parser.add_argument("--env_name", type=str, help="environment name")
    parser.add_argument(
        "--keys",
        type=str,
        action="append",
        default=None,
        help="environment physics key",
    )
    parser.add_argument(
        "--values",
        type=int,
        action="append",
        default=None,
        help="environment physics value",
    )
    parser.add_argument("--expert_model", type=str, help="expert model id")
    parser.add_argument("--horizon", type=int, default=10, help="mpc horizon")
    parser.add_argument("--history", type=int, default=5, help="mpc history")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="number of total epochs"
    )
    parser.add_argument(
        "--num_eval", type=int, default=3, help="number of evaluations"
    )
    parser.add_argument(
        "--save_video", type=bool, default=True, help="save one video"
    )
    return parser
