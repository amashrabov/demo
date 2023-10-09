import click

from .util import wd_path, check_name
from .launch import Launch

from .cfg import AppConfig
from .experiment.builder import build_all_experiment_actions

from .ci import cli as ci_cli
from higgsfield.internal.init import init


@click.command("run")
@click.option("--experiment_name", type=str, help="experiment name")
@click.option("--run_name", type=str, help="run name")
@click.option("--max_repeats", type=int, help="max repeats")
@click.argument("extra_args", nargs=-1)
def run_experiment(
    experiment_name: str,
    run_name: str,
    max_repeats: int,
    extra_args: list[str],
):
    wd = wd_path()
    app_config = AppConfig.from_path(wd)
    Launch(wd, app_config.name, experiment_name, run_name, max_repeats, extra_args)


@click.command("init")
@click.argument("project_name", type=str, required=True)
def init_cmd(project_name: str):
    print("Initializing {} project at:\n{}".format(project_name, wd_path()))
    init(wd_path(), check_name(project_name))


@click.command("update")
def update_experiment():
    wd = wd_path()
    app_config = AppConfig.from_path(wd)
    build_all_experiment_actions(wd, app_config.name)


@click.group("ci")
def ci():
    pass


ci.add_command(ci_cli.ssh_details)
ci.add_command(ci_cli.decode_secrets)
