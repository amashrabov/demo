from .params import Param, build_gh_action_inputs, build_run_params
from jinja2 import Environment, FileSystemLoader, Template
from higgsfield.internal.util import templates_path
from pathlib import Path
from importlib.machinery import SourceFileLoader
from .decorator import ExperimentDecorator
from higgsfield.internal.experiment.ast_parser import parse_experiments

header = """# THIS FILE WAS GENERATED BY HIGGSFIELD. 
# DO NOT EDIT. 
# IF YOUR WORKFLOW DOESN'T WORK, CREATE AN ISSUE.
"""


class ExperimentActionBuilder:
    project_name: str
    experiment_name: str
    params: list[Param]
    template: Template

    def __init__(self, project_name: str, experiment_name: str, params: list[Param]):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.params = params

        self.template = Environment(
            loader=FileSystemLoader(templates_path())
        ).get_template("experiment_action.j2")

    def render(self) -> str:
        return self.template.render(
            header=header,
            experiment_name=self.experiment_name,
            project_name=self.project_name,
            params=build_gh_action_inputs(self.params),
            rest=build_run_params(self.params),
        )


def _source_experiments(base_path: Path):
    """
    Only used inside docker to inject experiments into the module.
    Do not use outside of docker. But if you do, you will have to have
    the same dependencies (aka environment) as the docker container.
    """
    for file in base_path.glob("**/*.py"):
        try:
            SourceFileLoader("module.name", str(file)).load_module()
        except Exception as e:
            print(e)


def build_all_experiment_actions(root_path: Path, project_name: str):
    """
    Builds all experiment actions

    Root path should be the root of the project and must contain the src folder under which everything is defined.
    Project name should be the name of the project.

    It will parse ast of all files under src folder and search for experiments.
    Then it will build actions for each experiment and save them under .github/workflow folder.
    Deletes all actions that have name prefix run_experiment_ and header inside if not needed.
    """
    exp_params_pairs = []
    for file in (root_path / "src").glob("**/*.py"):
        exp_params_pairs.extend(parse_experiments(str(file.resolve())))

    if len(exp_params_pairs) == 0:
        print("No experiments found")
        return

    experiments = ExperimentDecorator.from_ast(exp_params_pairs)
    actions_folder = root_path / ".github" / "workflows"
    actions_folder.mkdir(parents=True, exist_ok=True)

    # list all files that have name prefix run_experiment_ and header inside
    for i in actions_folder.glob("run_experiment_*.yml"):
        if i.read_text().startswith(header):
            i.unlink()

    for experiment_name, experiment in experiments.items():
        action = ExperimentActionBuilder(
            project_name=project_name,
            experiment_name=experiment_name,
            params=experiment.params,
        ).render()

        action_file = actions_folder / f"run_experiment_{experiment_name}.yml"
        action_file.write_text(action)

        print("Updated experiment action", experiment_name)
