import json
import re
import click
import dotenv

from higgsfield.internal.util import wd_path
from higgsfield.internal.cfg import AppConfig, get_key_from_path_or_key
from .setup import Setup
from base64 import b64encode, b64decode


@click.command("get-hosts")
def hosts():
    wd = wd_path()
    app_config = AppConfig.from_path(wd)
    click.echo(",".join(app_config.hosts))


@click.command("get-nproc-per-node")
def proc_per_node():
    wd = wd_path()
    app_config = AppConfig.from_path(wd)
    click.echo(str(app_config.number_of_processes_per_node))


@click.command("get-ssh-details")
def ssh_details():
    wd = wd_path()
    app_config = AppConfig.from_path(wd)
    print(
        json.dumps(
            {
                "key": app_config.key,
                "user": app_config.user,
                "port": app_config.port,
                "hosts": ",".join(app_config.hosts),
            },
            indent=2,
        )
    )


@click.command("encode-secrets")
def encode_secrets():
    env_path = wd_path() / "env"
    if not env_path.exists():
        raise ValueError("env file not found")

    env: str | None = None
    with open(env_path, "r", encoding="UTF-8") as f:
        env = f.read()

    if env is None or len(env) == 0:
        raise ValueError("env file is empty")

    values = dotenv.dotenv_values(env_path)
    if "SSH_KEY" in values:
        values["SSH_KEY"] = get_key_from_path_or_key(values["SSH_KEY"])

    transform = lambda v: "'{}'".format(v.replace("'", "\\'"))
    env = "\n".join([f"{k}={transform(v)}" for k, v in values.items()])

    click.echo(b64encode(env.encode()).decode())


@click.command("decode-secrets")
@click.argument("env", type=str, required=True)
def decode_secrets(env: str):
    env_path = wd_path() / "env"
    if env_path.exists():
        raise ValueError("env file already exists")

    env_path.write_text(b64decode(env.encode()).decode())


https_repo_url_pattern = re.compile(
    r"^https\:\/\/github\.com\/[a-zA-Z0-9\-\_]+\/[a-zA-Z0-9\-\_]+\.git$"
)


def match_https_link(link: str) -> bool:
    return https_repo_url_pattern.match(link) is not None


def convert_https_to_ssh(link: str) -> str:
    gh, user, repo = link[8:-4].split("/")
    return f"git@{gh}:{user}/{repo}.git"


def parse_origin_link_or_else(link: str) -> str | None:
    if match_https_link(link):
        return convert_https_to_ssh(link)
    if link.startswith("git@github.com:"):
        return link

    return None


@click.command("setup-nodes")
def setup_nodes():
    wd = wd_path()
    app_config = AppConfig.from_path(wd)

    project_path = wd

    origin_url = app_config.get_git_origin_url(project_path)

    if origin_url is None:
        raise ValueError("Have you pushed your project to github?")

    origin_url = parse_origin_link_or_else(origin_url)

    if origin_url is None:
        raise ValueError("Please use ssh or https url for github repo.")

    app_config.github_repo_url = origin_url

    app_config.set_git_origin_url(project_path)

    setup = Setup(app_config, project_path)

    try:
        setup.create_ssh_key_file()
        setup.setup_nodes()
        setup.generate_deploy_action()
    finally:
        setup.finish()
