import os


def get_file_path_dataset(set_name="test"):
    """
    Find the accurate path for loading files of tests
    :return: correct file paths
    """

    filepath = "../imputegap/dataset/" + set_name + ".txt"

    if not os.path.exists(filepath):
        filepath = filepath[1:]

    return filepath


def get_save_path_asset():
    """
    Find the accurate path for saving files of tests
    :return: correct file paths
    """
    filepath = "../tests/assets"

    if not os.path.exists(filepath):
        filepath = filepath[1:]

    return filepath
