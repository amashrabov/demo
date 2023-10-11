from typing import List

import click

from .util import wd_path, check_name
from .launch import Launch

from .cfg import AppConfig
from .experiment.builder import DeployBuilder, build_all_experiment_actions

from .ci import cli as ci_cli
from higgsfield.internal.init import init

import os


def setup_environ_flags(rank):
    """Environment flags for debugging purposes"""

    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)


def setup(seed):
    import torch
    import torch.distributed as dist

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        setup_environ_flags(rank)


@click.command("run")
@click.option("--experiment_name", type=str, help="experiment name")
@click.option("--run_name", type=str, help="run name")
@click.option("--max_repeats", type=int, help="max repeats")
@click.argument("extra_args", nargs=-1)
def run_experiment(
    experiment_name: str,
    run_name: str,
    max_repeats: int,
    extra_args: List[str],
):
    wd = wd_path()
    app_config = AppConfig.from_path(wd)
    import os

    os.environ["PROJECT_NAME"] = app_config.name
    os.environ["EXPERIMENT_NAME"] = experiment_name
    os.environ["RUN_NAME"] = run_name
    setup(42)
    Launch(wd, app_config.name, experiment_name, run_name, max_repeats, extra_args)


@click.command("init")
@click.argument("project_name", type=str, required=True)
def init_cmd(project_name: str):
    print(
        "Initializing {} project at:\n{}/{}".format(
            project_name, wd_path(), project_name
        )
    )
    init(wd_path(), check_name(project_name))
    print("\nNow run:\n\n\tcd {}\n".format(project_name))


@click.command("build-experiments")
def update_experiment():
    wd = wd_path()
    app_config = AppConfig.from_path(wd)
    app_config.github_repo_url = app_config.get_git_origin_url(wd)
    app_config.set_git_origin_url(wd)
    DeployBuilder(app_config, wd).generate()
    build_all_experiment_actions(wd, app_config.name)


@click.group("ci")
def ci():
    pass

ci.add_command(ci_cli.proc_per_node)
ci.add_command(ci_cli.ssh_details)
ci.add_command(ci_cli.decode_secrets)
