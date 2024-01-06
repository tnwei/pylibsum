#!/usr/bin/env python3
from collections import OrderedDict
from pathlib import Path
import sys
import argparse

from .parser import count_libs

def main():
    ## Arugment handling
    parser = argparse.ArgumentParser(
        description="Summarizes libraries used in a Python script/repo"
    )
    parser.add_argument(
        "path", nargs="*", help="Path to file / folders to inspect", default=None
    )
    parser.add_argument(
        "-l",
        "--long",
        help="Shows results on individual file basis, else sums across files",
        action="store_true",
    )

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
                {
                    i: j
                    for i, j in sorted(
                        agg_libs.items(), key=lambda x: x[1], reverse=True
                    )
                }
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
                    {
                        i: j
                        for i, j in sorted(
                            file_lib.items(), key=lambda x: x[1], reverse=True
                        )
                    }
                )
                total_calls = sum(sorted_res.values())
                final_count = {
                    i: (j * 100 / total_calls) for i, j in sorted_res.items()
                }

                # Print to std out
                for i, j in final_count.items():
                    print(f"{i}: {j:.2f} %")

                # Leave a blank line as courtesy
                print()

        # Exit when done with loop
        sys.exit()


if __name__ == "__main__":
    main()
