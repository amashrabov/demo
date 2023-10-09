from typing import Tuple, List, Optional
from pydantic import BaseModel
from pathlib import Path
from importlib.machinery import SourceFileLoader
import os
import dotenv

import subprocess


def get_key_from_path_or_key(key_or_path: Optional[str]) -> str:
    if key_or_path is None or key_or_path == "":
        raise ValueError("SSH_KEY in env is None")

    path = Path(os.path.expanduser(key_or_path)).resolve()
    if path.exists():
        return path.read_text()

    return key_or_path


class AppConfig(BaseModel):
    name: str
    github_repo_url: Optional[str] = None
    hosts: List[str]
    user: str
    key: str 
    port: int
    number_of_processes_per_node: int

    @classmethod
    def from_path(cls, path: Path) -> "AppConfig":
        config_path = path / "src" / "config.py"
        if not config_path.exists():
            raise ValueError(f"Config file {config_path} not found")

        if not (path / "env").exists():
            raise ValueError(f"Env file {path/ 'env'} not found")
        dotenv.load_dotenv(path / "env", verbose=True, override=True)

        try:
            module = SourceFileLoader("module.name", str(config_path)).load_module()
        except Exception as e:
            raise ValueError(
                f"Config file {config_path} cannot be loaded since your file doesn't meet requirements"
            ) from e

        name = str(module.__dict__["NAME"])
        github_repo_url: str  = module.__dict__.get("GITHUB_REPO_URL", None)
        hosts = module.__dict__["HOSTS"]
        user = module.__dict__["HOSTS_USER"]
        port = module.__dict__["HOSTS_PORT"]
        number_of_processes_per_node = module.__dict__["NUMBER_OF_PROCESSES_PER_NODE"]

        key = get_key_from_path_or_key(os.getenv("SSH_KEY"))

        return AppConfig(
            name=name,
            github_repo_url=github_repo_url,
            hosts=hosts,
            user=user,
            key=key,
            port=port,
            number_of_processes_per_node=number_of_processes_per_node,
        )

    def get_git_origin_url(self, path) -> Optional[str]:
        if self.github_repo_url is not None:
            return self.github_repo_url
        try:
            # Run the Git command to get the remote origin URL
            result = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=path,
                universal_newlines=True,
            )

            # Strip any leading/trailing whitespace from the result
            origin_url = result.strip()

            return origin_url
        except subprocess.CalledProcessError:
            return None

    def set_git_origin_url(self, path: Path):
        config_path = path / "src" / "config.py"
        # remove the GITHUB_REPO_URL line
        with open(config_path, "r") as f:
            lines = f.readlines()
        with open(config_path, "w") as f:
            for line in lines:
                if line.find("GITHUB_REPO_URL") == -1:
                    f.write(line)
                else:
                    f.write(f'GITHUB_REPO_URL = "{self.github_repo_url}"\n')

    def is_valid(self) -> Optional[str]:
        if self.github_repo_url is None:
            return "GITHUB_REPO_URL is None"
        if self.key is None:
            return "SSH_KEY in env is None"
        return
