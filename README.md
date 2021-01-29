# pylibsum

<!-- Purpose of project -->
<!-- Brief project description -->

Command-line tool that gives a summary of libraries used in a Python script/repo, by giving a breakdown of functions called by proportion. The goal is to give a rough idea of what's in a script(s) or project folder, akin to Github's [Linguist](https://github.com/github/linguist/) for programming languages. 

Parses the AST of the code, and traces function calls to their imports. Hacked this together in a short amount of time, please pardon the rough edges. 

## Usage

``` bash
tnwei@rama:~$ pylibsum
Call signature: `pylibsum <INSERT DIRNAME>`
Example: Given contents of sample.py below:

	| import numpy as np
	| from plotnine import *
	| from sklearn.metrics import mean_squared_error
	| import scipy
	| a = np.array([1, 2, 3])
	| b = np.array([4, 5, 6])
	| c = b.mean()
	| mean_squared_error(a, b)
	| isinstance(10, list)
	| scipy.linalg.svd(a)

Outcome of running `pylibsum sample.py`:

Warning: wild card import found involving: `plotnine`
numpy: 50.00 %
sklearn.metrics: 16.67 %
scipy: 16.67 %
<built-in>: 16.67 %
<unknown>: 0.00 %
<user-defined>: 0.00 %
```

If multiple files are passed, their counts will be aggregated. Passing `-l` or `--long` will keep the counts separate. See below:

``` bash
# Without -l
tnwei@rama:~/test$ pylibsum .
<built-in>: 40.58 %
<unknown>: 28.99 %
<user-defined>: 8.70 %
pickle: 8.70 %
data: 2.90 %
subprocess: 2.90 %
xml.etree: 2.90 %
numpy: 1.45 %
pathlib: 1.45 %
tqdm: 1.45 %
os: 0.00 %

# With -l
tnwei@rama:~/test/$ pylibsum -l .
test_data.py

data: 100.00 %
<user-defined>: 0.00 %
<unknown>: 0.00 %
<built-in>: 0.00 %

data.py

<built-in>: 41.79 %
<unknown>: 29.85 %
pickle: 8.96 %
<user-defined>: 8.96 %
subprocess: 2.99 %
xml.etree: 2.99 %
numpy: 1.49 %
pathlib: 1.49 %
tqdm: 1.49 %
os: 0.00 %
```


## Installation

Clone this repo, then `cd` to it and run `pip install .`

## Requirements
<!-- Describe project requirements -->

No other packages required, standard library only. Python 3.6 and above.

------------------

Project based on the [cookiecutter-datascience-lite](https://github.com/tnwei/cookiecutter-datascience-lite/) template
