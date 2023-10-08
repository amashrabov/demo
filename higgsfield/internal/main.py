import click

from higgsfield.internal.cli import init_cmd, ci, run_experiment, update_experiment


@click.group()
def cli():
    """Higgsfield CLI"""
    pass


cli.add_command(init_cmd)
cli.add_command(ci)
cli.add_command(run_experiment)
cli.add_command(update_experiment)
