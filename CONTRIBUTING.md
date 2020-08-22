# Table of contents
*   [Contributing to Enchanter](#Contributing-to-Enchanter)
*   [Developing Enchanter](#Developing-Enchanter)
*   [Writing documentation](#Writing-documentation)

## Contributing to Enchanter
If you are interested in contributing to Enchanter, your contributions will fall into two categories:

1.  You want to propose a new feature and implement it.
Post about your intended feature, and we shall discuss the design and implementation. 
Once we agree that the plan looks good, go ahead and implement it.

2.  You want to implement a feature or bug-fix for an outstanding issue.
    *   Search for your issue here: [https://github.com/khirotaka/enchanter/issues](https://github.com/khirotaka/enchanter/issues)
    *   Pick an issue and comment on the task that you want to work on this feature.
    *   If you need more context on a particular issue, please ask and we shall provide.

## Developing Enchanter
To develop Enchanter on your machine.

1.  Install Poetry

    Enchanter uses [Poetry](https://python-poetry.org) and `pyproject.toml` to manage dependencies.  
    Please refer to [the official documentation](https://python-poetry.org/docs/#installation) and install the appropriate one for your platform.

2.  Clone a copy of Enchanter from source:
    ```shell script
    git clone https://github.com/khirotaka/enchanter.git
    cd enchanter
    ```

    1.  If you already have Enchanter from source, update it:
    ```shell script
    git pull --rebase
    git submodule sync --recursive
    git submodule update --init --recursive
    ```

    and install Enchanter on develop mode.

    ```shell script
    poetry develop    # install enchanter
    poetry shell      # activate virtual environment
    ```

## Writing documentation
Enchanter uses Google style for formatting docstrings. 

### Building documentation
To build the documentation

1.  Build and install Enchanter
2.  Install the requirements
    ```shell script
    cd docs/
    pip install -r requirements.txt
    ```
    
3.  Generate the documentation HTML files.

    ```shell script
    make html
    ```
