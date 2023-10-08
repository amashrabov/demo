import ast

from typing import Any


def func_defs(module: ast.Module) -> list[ast.FunctionDef]:
    defs = []
    for node in ast.iter_child_nodes(module):
        if isinstance(node, ast.FunctionDef):
            defs.append(node)

    return defs


def filter_experiment_defs(defs: list[ast.FunctionDef]):
    filtered_defs: list[ast.FunctionDef] = []
    for node in defs:
        try:
            if (
                len(node.decorator_list) >= 1
                and node.decorator_list[0].func.id == "experiment"  # type: ignore
            ):
                filtered_defs.append(node)
        except Exception:
            pass
    return filtered_defs


class Dec:
    name: str
    arg_pairs: dict[str, Any]
    allowed_args = dict()

    def __init__(
        self,
        name: str,
        allowed_args: dict[str, tuple[type, ...]],
        arg_pairs: dict[str, Any] | None = None,
    ):
        self.name = name
        self.allowed_args = allowed_args
        self.arg_pairs = dict() if arg_pairs is None else arg_pairs

    def add_arg_pair(self, left: str, right: Any):
        if left not in self.allowed_args:
            raise ValueError(f"argument {left} of {self.name} is not allowed")
        if type(right) not in self.allowed_args[left]:
            raise ValueError(
                f"argument {left} of {self.name} has type {type(right)}, need {self.allowed_args[left]}"
            )

        if left in self.arg_pairs:
            raise ValueError(f"argument {left} is redefined in {self.name}")

        self.arg_pairs[left] = right


noneType = type(None)


class Expdec(Dec):
    def __init__(self):
        super(Expdec, self).__init__(
            name="experiment", allowed_args={"name": (str,), "seed": (int, noneType)}
        )


class Paramdec(Dec):
    def __init__(self):
        super(Paramdec, self).__init__(
            name="param",
            allowed_args={
                "name": (str,),
                "default": (str, int, bool, float, noneType),
                "description": (str, noneType),
                "required": (bool, noneType),
                "type": (type,),
                "options": (
                    tuple,
                    noneType,
                ),
            },
        )


builder = {"experiment": Expdec, "param": Paramdec}

type_dict = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def build_experiment_def(
    node: ast.FunctionDef,
) -> tuple[Expdec, dict[str, Paramdec]] | None:
    experiment: Expdec | None = None
    params: dict[str, Paramdec] = dict()
    stop = False

    for maybe_decorator in node.decorator_list:
        if not isinstance(maybe_decorator, ast.Call):
            continue
        decorator: ast.Call = maybe_decorator  # type: ignore
        func: ast.Name | None = getattr(decorator, "func", None)

        if func is None:
            stop = True
            break
        dec_to_call = builder.get(func.id, None)

        if dec_to_call is None:
            stop = True
            break

        dec = dec_to_call()

        if len(decorator.args) > 1:
            raise ValueError(
                "experiment or param decorators cannot "
                + "have other decorators applied or unnamed params other than name"
            )

        if len(decorator.args) == 1:
            try:
                dec.add_arg_pair("name", decorator.args[0].value)  # type: ignore
            except Exception:
                stop = True
                break

        for kw in decorator.keywords:
            try:
                field_val = None
                if isinstance(kw.value, ast.Name):
                    field_val = type_dict[kw.value.id]
                elif isinstance(kw.value, ast.Constant):
                    field_val = kw.value.value
                elif isinstance(kw.value, ast.Tuple) or isinstance(kw.value, ast.List):
                    vals = []
                    for elt in kw.value.elts:
                        vals.append(elt.value)  # type: ignore
                    field_val = tuple(vals)

                else:
                    raise ValueError(
                        f"cannot find the type of kw.value: {type(kw.value)}"
                    )
                dec.add_arg_pair(kw.arg, field_val)  # type: ignore
            except Exception as e:
                stop = True
                break

        if type(dec) == Expdec:
            if experiment is not None:
                raise ValueError("more than one experiment is defined")
            else:
                experiment = dec

        if type(dec) == Paramdec:
            if dec.arg_pairs["name"] in params:
                raise ValueError(
                    f'more than one param with the same name {dec.arg_pairs["name"]} is defined'
                )
            params[dec.arg_pairs["name"]] = dec
    if stop:
        return

    if experiment is None:
        return

    return experiment, params


def build_experiment_defs(
    defs: list[ast.FunctionDef],
) -> list[tuple[Expdec, dict[str, Paramdec]]]:
    exps: list[tuple[Expdec, dict[str, Paramdec]]] = list()
    for node in defs:
        ret = build_experiment_def(node)
        if ret is not None:
            exp, params = ret
            exps.append((exp, params))

    return exps


def parse_experiments(filename: str) -> list[tuple[Expdec, dict[str, Paramdec]]]:
    parsed_code: ast.Module | None = None
    with open(filename, "r") as f:
        parsed_code = ast.parse(f.read())

    if parsed_code == None:
        return []

    # search for top level experiment declarations
    defs = func_defs(parsed_code)

    # filter out by decorators, top level decorator should be "experiment"
    defs = filter_experiment_defs(defs)

    got = build_experiment_defs(defs)

    return got
