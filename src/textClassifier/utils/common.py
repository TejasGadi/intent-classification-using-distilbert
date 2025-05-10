import os
from box.exceptions import BoxValueError
import yaml
from src.textClassifier.logging import logger
from beartype import beartype
from box import ConfigBox
from pathlib import Path
from typing import Any

# Common functions that is needed and used throughout the projects

# Note: ensure_annotations decorator used so that it ensures the function parameters and output strictly typed, so while passing incorrect typed parameter it should throw an error(which is not achieved by just defining the paremeter type)

@beartype
def read_yaml(path_to_yaml:Path)-> ConfigBox:
    """
    Reads yaml file and returns

    Args:
        path_to_yaml (str) : path like input

    Raises:
        ValueError: if yaml file empty
        e: empty File

    Returns:
        ConfigBox: ConfigBox type
    """

    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} is loaded sucessfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file empty")
    except Exception as e:
        raise e
    
@beartype
def create_directories(path_to_directories: list[str], verbose:bool= True):
    """
    create list of directories

    Args:
        path_to_directories (list[Path]) : list of path directories
        verbose (bool, optional): Ignore log prints if verbose is false, defaults to true
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")