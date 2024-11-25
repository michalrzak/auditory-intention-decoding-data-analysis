# Auditory Intention Decoding data-analysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project was developed as part of my master thesis - _Towards an intention decoding auditory BCI_. The thesis aims
to investigate and lay the groundwork for a potential future brain-computer interface (BCI) paradigm, where a system
could decode an attended stimulus given sequentially presented auditory stimuli.

This repository contains the data-analysis code used to generate the results and figures presented in the thesis.

The experiment codebase can be found under: <https://github.com/michalrzak/auditory-intention-decoding>

## Abstract of master thesis
Auditory brain-computer interfaces (BCIs) prime users with auditory signals and observe their brain activity. These BCI types are relatively unexplored in literature even though we see great potential in the interaction they enable, especially if utilized in so-called reactive or passive BCI paradigms. In these, the user does not have to perform any explicit action, instead, the system infers information from the user's brain activity and controls the system implicitly.

This thesis introduces the idea and prototype of a reactive auditory BCI based on electroencephalography (EEG), termed auditory intention decoding (AID). The goal of AID is to infer the user's intention by probing them with possible words and observing their brain response. We present an idea where such a system could be used in the future as a conversational aid for speech-impaired individuals. 

The thesis further investigates modulating the options of AID with various techniques inspired by visually evoked potentials, termed taggers. With these, we hope to improve the decoding accuracy of AID, by highlighting involved brain regions in processing the words with simple-to-decode tags. 

Four subject recordings in addition to three internal recordings were conducted on a simplified version of the AID paradigm to determine its feasibility. The data analysis revealed a significantly decodable effect, though it is not strong enough to be used in a practical setting. Further, the subset of tested taggers could not reveal any advantage in decoding performance, suggesting the need for further refinement. 

While current results do not yet support AID's practical deployment, the findings provide a foundation for future research. Subsequent studies could explore untested tagging techniques or alternative auditory stimuli to improve accuracy, leveraging the methods and data established in this thesis to unlock AID's full potential.

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
