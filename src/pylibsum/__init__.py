#!/usr/bin/env python3
import ast
from typing import List, Dict
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
import sys
import argparse


# From https://docs.python.org/3/library/functions.html
# No idea how this changed over time
BUILTINS: List[str] = [
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

# Literals referred from https://greentreesnakes.readthedocs.io/en/latest/nodes.html#literals
AST_LITERALS = (
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
)


class ImportTracker(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.libs: Dict[str, str] = {}

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
                    print(f"Warning: wild card import found involving: `{node.module}`")
                    self.libs[name] = "<unknown>"
            else:
                self.libs[asname] = name
                self.libs[name] = node.module

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
                AST_LITERALS,
            ):
                return "<built-in>"
            else:
                # print(ast.dump(current_layer))
                # raise Exception
                return "<unknown>"


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
            # A bit tedious here as mapping out the source library for each
            # value assignment isn't straightforward
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
            elif isinstance(i, AST_LITERALS):
                # Nothing worth tracking if they're literals I think?
                # The sequence literals that can take any valid Python object
                # are near impossible to track here
                continue
            else:
                # TODO: Map out other conditional branches and remove this           continue
                # print(type(i))
                # raise Exception
                continue

            if isinstance(node.value, ast.Call):
                # If it's a function call, track the origin of the assigned object
                traced_origin = self.find_top_lvl_name(node.value.func)

                if name == traced_origin:
                    # The origin couldn't be determined
                    # Refrain from creating self-referential loop
                    pass
                else:
                    # Record the origin of the assigned object
                    self.assigns[name] = traced_origin

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
                AST_LITERALS,
            ):
                return "<built-in>"
            else:
                # print(ast.dump(current_layer))
                # raise Exception
                return "<unknown>"


class LibSum(
    ImportTracker, FunctionDefTracker, CallTracker, AssignTracker, ast.NodeVisitor
):
    # Lumping this here, so that each component is in its own class
    # TODO: Clean up
    pass


def count_libs(text):
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
    final_count["<user-defined>"] = 0
    final_count["<unknown>"] = 0
    final_count["<built-in>"] = 0

    # For the associated item in the list
    for i in obj.calls:
        # If it is an object, lookup to see if can assign the object to a library
        if i in obj.assigns.keys():
            j = i
            for _ in range(30):
                # Probably no such thing that is nested 30 layers deep?
                # Danger of a circular reference here
                if j in obj.assigns.keys():
                    j = obj.assigns.get(j)
                else:
                    i = j
                    break
            else:
                # If we exceed 30 loops, we probably have a circular reference
                print(f"Warning: Circular reference for {i}, assigning as unknown")
                final_count["<unknown>"] += 1
                continue

        # If it is a function, and is defined in code
        if i in obj.functiondefs:
            final_count["<user-defined>"] += 1

        # If it is a function, and can be directly traced to one of the libraries
        elif i in obj.libs.keys():
            final_count[obj.libs.get(i)] += 1

        # If it is one of the builtins
        elif i in BUILTINS:
            final_count["<built-in>"] += 1

        # Else, cannot trace lineage of function
        else:
            final_count["<unknown>"] += 1

    total_calls = sum(final_count.values())

    if total_calls == 0:
        # print("No functions called")
        return {}


    return final_count


def main():
    ## Arugment handling
    parser = argparse.ArgumentParser(
        description="Summarizes libraries used in a Python script/repo"
    )
    parser.add_argument(
        "path", nargs="*", help="Path to file / folders to inspect", default=None
    )
    parser.add_argument(
        "-l", "--long", help="Shows results on individual file basis, else sums across files", action="store_true")

    args = parser.parse_args()


    # If we have no dir names passed, display a meaningful example
    if len(args.path) == 0:
        print("Call signature: `pylibsum <INSERT DIRNAME>`")
        print("Example: Given contents of sample.py below:")
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
        print("Outcome of running `pylibsum sample.py`:")
        print()

        # This line does the heavy lifting
        res = count_libs(text)

        # These lines sort libraries by descending order
        # and prints them
        sorted_res = OrderedDict(
            {i: j for i, j in sorted(res.items(), key=lambda x: x[1], reverse=True)}
        )

        total_calls = sum(sorted_res.values())
        final_count = {i: (j * 100 / total_calls) for i, j in sorted_res.items()}
        for i, j in final_count.items():
            print(f"{i}: {j:.2f} %")

        print()

    # We have args passed
    else:
        # Vet to only include Python scripts
        fnames = []

        for i in args.path:
            fp = Path(i)
            if fp.is_dir():
                # Include *.py, *.pyi files
                fnames.extend(list(fp.glob("*.py")))
                fnames.extend(list(fp.glob("*.pyi")))
            elif fp.suffix == ".py":
                fnames.append(fp)
            else:
                pass

        file_libs = {}

        # Run counts for each file
        for fn in fnames:
            with open(fn, "r") as f:
                text = f.read()
            file_libs[fn] = count_libs(text)

        # If aggregating (default behaviour)
        if args.long is False:
            # Need to collate counts
            agg_libs = {}

            # Iterate across all stored counts
            for _, file_lib in file_libs.items():
                for lib, count in file_lib.items():
                    if lib not in agg_libs.keys():
                        # Add to aggregate dict if lib not recorded
                        agg_libs[lib] = count
                    else:
                        # Add to existing count if lib is recorded
                        agg_libs[lib] += count

            # Sort by descending count
            sorted_res = OrderedDict(
                {i: j for i, j in sorted(agg_libs.items(), key=lambda x: x[1], reverse=True)}
            )
            total_calls = sum(sorted_res.values())
            final_count = {i: (j * 100 / total_calls) for i, j in sorted_res.items()}

            # Print to stdout
            for i, j in final_count.items():
                print(f"{i}: {j:.2f} %")

            # Leave a blank line as courtesy
            print()

        # If not aggregating, and reporting by file
        else:
            # Iterate for each file
            for fn, file_lib in file_libs.items():
                # Print the file name
                print(fn)
                print()

                # Sort by descending count
                sorted_res = OrderedDict(
                    {i: j for i, j in sorted(file_lib.items(), key=lambda x: x[1], reverse=True)}
                )
                total_calls = sum(sorted_res.values())
                final_count = {i: (j * 100 / total_calls) for i, j in sorted_res.items()}

                # Print to std out
                for i, j in final_count.items():
                    print(f"{i}: {j:.2f} %")

                # Leave a blank line as courtesy
                print()

        # Exit when done with loop 
        sys.exit()


if __name__ == "__main__":
    main()
