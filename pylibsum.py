import ast
from typing import List, Dict
from collections import OrderedDict

# From https://docs.python.org/3/library/functions.html
# No idea how this changed over time
builtin_functions: List[str] = [
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "breakpoint",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "compile",
    "complex",
    "delattr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "eval",
    "exec",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "globals",
    "hasattr",
    "hash",
    "help",
    "hex",
    "id",
    "input",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "locals",
    "map",
    "max",
    "memoryview",
    "min",
    "next",
    "object",
    "oct",
    "open",
    "ord",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
    "__import__",
]


class ImportTracker(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.libs: Dict[str] = {}

    def visit_Import(self, node):
        """
        node.names are basically what is imported
        node.module is where they are from

        The idea is to have self.libs be a lookup, so that
        the modules where functions can be imported from can
        be traced.
        """
        for i in node.names:
            assert isinstance(i, ast.alias)
            name = i.name
            asname = i.asname

            if asname is None:
                self.libs[name] = name
            else:
                self.libs[asname] = name

        # Call self.generic_visit(node) to include child nodes
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for i in node.names:
            assert isinstance(i, ast.alias)
            name = i.name
            asname = i.asname

            if asname is None:
                if name != "*":
                    self.libs[name] = node.module
                else:
                    print(f"Warning: `from {node.module} import *` detected!")
                    self.libs[name] = "unknown"
            else:
                self.libs[asname] = {name: node.module}

        # Call self.generic_visit(node) to include child nodes
        self.generic_visit(node)


class FunctionDefTracker(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.functiondefs: List[str] = []

    def visit_FunctionDef(self, node):
        """
        node.name is the function name. Just have to track that.
        """
        self.functiondefs.append(node.name)

        # Call self.generic_visit(node) to include child nodes
        self.generic_visit(node)


class CallTracker(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.calls: List[str] = []

    def visit_Call(self, node):
        # Take the node.func object, which is either ast.Name or ast.Attribute
        if isinstance(node.func, ast.Name):
            # This is if the function call is a single line
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # This is if the function call has multiple submodules
            # Find the top-level name and store it!
            # TODO
            toplvlname = self.find_top_lvl_name(node.func)
            self.calls.append(toplvlname)
        else:
            pass

        # Call self.generic_visit(node) to include child nodes
        self.generic_visit(node)

    def find_top_lvl_name(self, func):
        # Wade through the first ast.Attribute of each layer until an ast.Name is found
        current_layer = func
        for _ in range(10):  # no such thing as 10 nested attributes!
            if isinstance(current_layer, ast.Name):
                return current_layer.id
            elif isinstance(current_layer, ast.Attribute):
                current_layer = current_layer.value
            elif isinstance(current_layer, ast.Call):
                # If it's a Call, we'll get to it eventually
                pass
            elif isinstance(current_layer, ast.Subscript):
                current_layer = current_layer.value
            elif isinstance(current_layer, ast.BinOp):
                # Choose left as a guess by human writing convention
                current_layer = current_layer.left
            elif isinstance(
                current_layer,
                (
                    ast.Constant,
                    ast.Str,
                    ast.Num,
                    ast.Str,
                    ast.FormattedValue,
                    ast.JoinedStr,
                    ast.Bytes,
                    ast.List,
                    ast.Tuple,
                    ast.Set,
                    ast.Dict,
                    ast.Ellipsis,
                    ast.NameConstant,
                ),
            ):
                # Literals, any function from these are builtins I believe?
                # Literals referred from https://greentreesnakes.readthedocs.io/en/latest/nodes.html#literals
                return "builtin"
            else:
                print(ast.dump(current_layer))
                raise Exception


class AssignTracker(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.assigns: Dict[str] = {}

    def visit_Assign(self, node):
        # In an ast.Assign, we have `targets` as a list of node, and `value` as a single node
        # Most likely that `targets` contains ast.Names, and `value` contains Calls.
        # Isn't always true though!
        # assert isinstance(node.value, ast.Call), f"{ast.dump(node.value)}"

        for i in node.targets:
            if isinstance(i, ast.Name):
                name = i.id
            elif isinstance(i, ast.Attribute):
                name = i.value
            elif isinstance(i, ast.Subscript):
                name = i.value
            elif isinstance(i, ast.BinOp):
                # Can't tell what's the output of the BinOp w/o evaluating it!
                # Usually these are numbers, but pathlib.Path() for example
                # overloads division to concat subdirectories
                continue
            else:
                # TODO: Map out other conditional branches and remove this
                print(type(i))
                raise Exception

            if isinstance(node.value, ast.Call):
                # If it's a function call, track the origin of the function
                self.assigns[name] = self.find_top_lvl_name(node.value.func)
            else:
                # If it isn't, try and find out the origin
                # TODO: Find the origin of the variables assigned
                pass

        # Call self.generic_visit(node) to include child nodes
        self.generic_visit(node)

    # TODO: Merge duplicated with above
    # TODO: This is not limited to `ast.Call.func` innit? Can be made generic
    def find_top_lvl_name(self, func):
        # Wade through the first ast.Attribute of each layer until an ast.Name is found
        current_layer = func
        for _ in range(10):  # no such thing as 10 nested attributes!
            if isinstance(current_layer, ast.Name):
                return current_layer.id
            elif isinstance(current_layer, ast.Attribute):
                current_layer = current_layer.value
            elif isinstance(current_layer, ast.Call):
                # If it's a Call, we'll get to it eventually
                pass
            elif isinstance(current_layer, ast.Subscript):
                current_layer = current_layer.value
            elif isinstance(current_layer, ast.BinOp):
                # Choose left as a guess by human writing convention
                current_layer = current_layer.left
            elif isinstance(
                current_layer,
                (
                    ast.Constant,
                    ast.Str,
                    ast.Num,
                    ast.Str,
                    ast.FormattedValue,
                    ast.JoinedStr,
                    ast.Bytes,
                    ast.List,
                    ast.Tuple,
                    ast.Set,
                    ast.Dict,
                    ast.Ellipsis,
                    ast.NameConstant,
                ),
            ):
                # Literals, any function from these are builtins I believe?
                # Literals referred from https://greentreesnakes.readthedocs.io/en/latest/nodes.html#literals
                return "builtin"
            else:
                print(ast.dump(current_layer))
                raise Exception


class LibSum(
    ImportTracker, FunctionDefTracker, CallTracker, AssignTracker, ast.NodeVisitor
):
    pass


def count_libs(text, return_percent=True):
    tree: ast.Module = ast.parse(text)
    obj: LibSum = LibSum()
    obj.visit(tree)

    # The 4 lists that we end up with
    # obj.assigns
    # obj.calls
    # obj.functiondefs
    # obj.libs

    # Pre-populate
    final_count: Dict[int] = {i: 0 for i in obj.libs.values()}
    final_count["user-defined"] = 0
    final_count["unknown"] = 0
    final_count["builtin"] = 0

    # For the associated item in the list
    for i in obj.calls:
        # If it is an object, lookup to see if can assign the object to a library
        if i in obj.assigns.keys():
            j = i
            while True:
                if j in obj.assigns.keys():
                    j = obj.assigns.get(j)
                else:
                    break
            i = j

        # If it is a function, and is defined in code
        if i in obj.functiondefs:
            final_count["user-defined"] += 1

        # If it is a function, and can be directly traced to one of the libraries
        elif i in obj.libs.keys():
            final_count[obj.libs.get(i)] += 1

        # If it is one of the builtins
        elif i in builtin_functions:
            final_count["builtin"] += 1

        # Else, cannot trace lineage of function
        else:
            final_count["unknown"] += 1

    if return_percent is True:
        total_calls = sum(final_count.values())
        final_count = {i: (j * 100 / total_calls) for i, j in final_count.items()}

    return final_count


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Show proportion of functions called by library"
    )
    parser.add_argument("path", nargs="?", help="Path to file to inspect", default=None)

    args = parser.parse_args()

    if args.path is None:
        print("Call signature: `python pylibsum.py <INSERT FILENAME>`")
        print("Example: Given contents of example.py below:")
        text = """
import numpy as np
from plotnine import *
from sklearn.metrics import mean_squared_error
import scipy
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = b.mean()
mean_squared_error(a, b)
isinstance(10, list)
scipy.linalg.svd(a)"""
        print("\n\t| ".join(text.split("\n")))
        print()
        print("Outcome of running `python pylibsum.py example.py`:")
        print()
        res = count_libs(text)
        sorted_res = OrderedDict(
            {i: j for i, j in sorted(res.items(), key=lambda x: x[1], reverse=True)}
        )
        for i, j in sorted_res.items():
            print(f"{i}: {j:.2f} %")
        sys.exit()
    else:
        fn = args.path

    with open(fn, "r") as f:
        text = f.read()

    # Nice thing w/ AST is that it ignores comments!
    # The below do not show up
    # import antigravity
    # print(astpp.dump(ast.parse(text)))
    # print()
    res = count_libs(text)
    sorted_res = OrderedDict(
        {i: j for i, j in sorted(res.items(), key=lambda x: x[1], reverse=True)}
    )
    for i, j in sorted_res.items():
        print(f"{i}: {j:.2f} %")
