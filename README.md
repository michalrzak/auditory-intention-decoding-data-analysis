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

Despite its significant potential, the auditory brain-computer interface (BCI) field remains underexplored. Current paradigms predominantly focus on decoding which audio source among several a user is attending to, limiting their applicability in real-world scenarios. This thesis addresses this gap by conceptualizing and testing a novel BCI paradigm centered around auditory stimulation. Our approach, Auditory Intention Decoding (AID), is classified as a reactive BCI, relying on automatic brain responses rather than direct user input. The AID paradigm aims to infer a user's intended choice among multiple presented options, from the user's automatic brain response, without them needing to take any explicit action.

Additionally, we investigate the effectiveness of modulating the auditory stimuli with distinct waveforms, akin to techniques used in visually evoked potentials, to enhance target decoding accuracy. These concepts were empirically tested through a controlled experiment, and the results, while showing a discernible effect, suggest that the paradigm in its current form is not yet robust enough for practical application. Nevertheless, the findings presented here lay a foundation for future research, offering a promising direction for developing more useful auditory BCIs

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
