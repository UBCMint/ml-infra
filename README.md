# MINT MOSS Infrastructure and MLOps 

Developing infrastructure and MLOps tools for MINT MOSS.

## Setup and Installation
### Prerequisites
* Python 3.12
* Latest [Anaconda Distribution](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) Installed
    * **NOTE:** If unsure which one to install, see [this](https://docs.anaconda.com/distro-or-miniconda/).

### Set up Workspace

#### Fork this repository:

* Navigate to the GitHub repository.
* Click on the "Fork" button in the top-right corner.
* Clone the forked repository to your local machine and change working directory:
 ```
git clone https://github.com/UBCMint/ml-infra.git
cd ml-infra
```

#### Create and Activate Conda Environment

```
conda create -p venv python=3.12 -y
conda activate venv/
```

#### Install the required packages and dependencies:

```
pip install -r requirements.txt
```
