# Auditory Intention Decoding data-analysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project was developed as part of my master thesis - _Towards an intention decoding auditory BCI_. The thesis aims
to investigate and lay the groundwork for a potential future brain-computer interface (BCI) paradigm, where a system
could decode an attended stimulus given sequentially presented auditory stimuli.

This repository contains the data-analysis code used to generate the results and figures presented in the thesis.

The experiment codebase can be found under: <https://github.com/michalrzak/auditory-intention-decoding>

## Abstract
[TBD]

## Running the codebase

The repository is based around a Makefile, so make is required to reproduce the results (this can be a bit tedious on
windows).

### Installing the requirements

The requirements of this repository are managed with the tool [Poetry](https://python-poetry.org/), hence it needs to be
installed before any other steps can be taken.

After installing Poetry, you can run the following make rule to create an environment with all dependencies:

```
make requirements
```

**Note**: I prefer to have my virtual environment saved in the repository folder, which is not the default behavior of
Poetry. To change this, follow the instructions outlined
in: <https://python-poetry.org/docs/configuration/#virtualenvsin-project>.

### Generating all results

After installing the requirements use the following make rule to generate all plots used throughout the thesis. Note
that this will take a _VERY_ long time to run.

```
make all
```

### Generating results one-by-one

The results can also be generated one-by-one using their individual make rules. To get a list of all available rules
run:

```
make
```

The command will list all possible make rules along with a short description of what each rule generates
