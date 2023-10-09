from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from higgsfield.internal.cfg import AppConfig
from fabric import Connection, SerialGroup

from higgsfield.internal.util import templates_path
from higgsfield.internal.experiment.builder import header


class Setup:
    app_config: AppConfig
    path: str
    deploy_key: str
    project_path: Path

    def __init__(
        self,
        app_config: AppConfig,
        project_path: Path,
    ):
        self.app_config = app_config

        if reason := self.app_config.is_valid() is not None:
            raise ValueError(reason)

        self.project_path = project_path

    def create_ssh_key_file(self):
        if self.app_config.key is None:
            raise ValueError("SSH_KEY in env is None")

        with Path.home() / ".ssh" / f"{self.app_config.name}.key" as f:
            f.write_text(self.app_config.key)
            f.chmod(0o400)
            self.path = str(Path.resolve(f.absolute()))

    def finish(self):
        Path(self.path).unlink()

    def establish_connections(self):
        if self.app_config.key is None:
            raise ValueError("SSH_KEY in env is None")

        self.connections = [
            Connection(
                host=host,
                port=self.app_config.port,
                user=self.app_config.user,
                connect_kwargs={"key_filename": self.path},
            )
            for host in self.app_config.hosts
        ]
        self.group = SerialGroup.from_connections(self.connections)

    def set_deploy_key(self):
        with Path.home() / ".ssh" / "higgsfield" / f"{self.app_config.name}-github-deploy.key" as f:
            self.deploy_key = f.read_text()

    def _build_deploy_key_string(self):
        return f"Host github.com-{self.app_config.name}\n\tHostName github.com\n\tIdentityFile ~/.ssh/{self.app_config.name}-github-deploy.key\n\tIdentitiesOnly yes\n\tStrictHostKeyChecking no\n\tUserKnownHostsFile=/dev/null\n\tLogLevel=ERROR\n"

    def setup_nodes(self):
        self.establish_connections()

        # ssh into each node and install docker
        install_docker_script = '''/bin/bash -c "$(curl -fsSL https://gist.githubusercontent.com/arpanetus/1c1210b9e432a04dcfb494725a407a70/raw/5d47baa19b7100261a2368a43ace610528e0dfa2/install.sh)"'''
        self.group.run(install_docker_script)

        # ssh into each node and put deploy key for repo
        self.group.run(f"mkdir -p ~/.ssh")

        self.set_deploy_key()

        # we need to check first if the key is already there
        # if it is, we need to remove it
        self.group.run(f"rm -f ~/.ssh/{self.app_config.name}-github-deploy.key || true")

        self.group.run(
            f"echo '{self.deploy_key}' > ~/.ssh/{self.app_config.name}-github-deploy.key"
        )
        self.group.run(f"chmod 400 ~/.ssh/{self.app_config.name}-github-deploy.key")
        self.group.run(f"echo '{self._build_deploy_key_string()}' >> ~/.ssh/config")

        self.group.run(
            """wget https://github.com/ml-doom/invoker/releases/download/latest/invoker-latest-linux-amd64.tar.gz && \
        tar -xvf invoker-latest-linux-amd64.tar.gz && \
        sudo mv invoker /usr/bin/invoker && \
        rm invoker-latest-linux-amd64.tar.gz"""
        )

    def generate_deploy_action(self):
        path = self.project_path / ".github" / "workflows" / "deploy.yml"

        template = Environment(loader=FileSystemLoader(templates_path())).get_template(
            "deploy_action.j2"
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        keyed_repo_url = self.app_config.github_repo_url
        assert keyed_repo_url is not None
        keyed_repo_url.replace("github.com/", f"github.com-{self.app_config.name}/")

        path.write_text(
            template.render(
                header=header,
                project_name=self.app_config.name,
                keyed_repo_url=keyed_repo_url,
            )
        )
