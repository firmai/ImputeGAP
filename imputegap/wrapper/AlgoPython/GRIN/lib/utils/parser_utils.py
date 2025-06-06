# ===============================================================================================================
# SOURCE: https://github.com/Graph-Machine-Learning-Group/grin
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://openreview.net/pdf?id=kOu3-S3wJ7
# ===============================================================================================================

import inspect
from argparse import Namespace, ArgumentParser
from typing import Union


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n', 'off'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y', 'on'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


from argparse import Namespace


def config_dict_from_args(args: Namespace) -> dict:
    """
    Extracts a dictionary with the experiment configuration from arguments
    (necessary to filter out TestTube arguments).

    :param args: Namespace (e.g., parsed arguments)
    :return: Dictionary containing filtered hyperparameters
    """
    keys_to_remove = {
        'hpc_exp_number', 'trials', 'optimize_parallel', 'optimize_parallel_gpu',
        'optimize_parallel_cpu', 'generate_trials', 'optimize_trials_parallel_gpu'
    }

    # Convert Namespace to dictionary if necessary
    if isinstance(args, Namespace):
        args_dict = vars(args)
    else:
        args_dict = args  # If already a dictionary, use it directly

    # Filter out unwanted keys
    hparams = {key: v for key, v in args_dict.items() if key not in keys_to_remove}
    return hparams


def update_from_config(args: Namespace, config: dict) -> Namespace:
    """
    Updates an args Namespace object with values from a given configuration dictionary.

    :param args: Namespace object containing arguments
    :param config: Dictionary with updated parameters
    :return: Updated args Namespace object
    """
    # Ensure config keys exist in args before updating
    missing_keys = set(config.keys()).difference(vars(args))
    if missing_keys:
        raise ValueError(f"Keys {missing_keys} not found in args.")

    args_dict = vars(args)
    args_dict.update(config)  # Update only existing keys
    return args

def parse_by_group(parser):
    """
    Create a nested namespace using the groups defined in the argument parser.
    Adapted from https://stackoverflow.com/a/56631542/6524027

    :param args: arguments
    :param parser: the parser
    :return:
    """
    assert isinstance(parser, ArgumentParser)
    args = parser.parse_args()

    # the first two argument groups are 'positional_arguments' and 'optional_arguments'
    pos_group, optional_group = parser._action_groups[0], parser._action_groups[1]
    args_dict = args._get_kwargs()
    pos_optional_arg_names = [arg.dest for arg in pos_group._group_actions] + [arg.dest for arg in
                                                                               optional_group._group_actions]
    pos_optional_args = {name: value for name, value in args_dict if name in pos_optional_arg_names}
    other_group_args = dict()

    # If there are additional argument groups, add them as nested namespaces
    if len(parser._action_groups) > 2:
        for group in parser._action_groups[2:]:
            group_arg_names = [arg.dest for arg in group._group_actions]
            other_group_args[group.title] = Namespace(
                **{name: value for name, value in args_dict if name in group_arg_names})

    # combine the positiona/optional args and the group args
    combined_args = pos_optional_args
    combined_args.update(other_group_args)
    return Namespace(flat=args, **combined_args)


import inspect
from argparse import Namespace
from typing import Union

def filter_args(args: Union[Namespace, dict], target_cls, return_dict=False):
    """
    Filters the given args to only include the arguments required by the target class.

    :param args: A Namespace or dictionary of arguments.
    :param target_cls: The class whose arguments should be filtered.
    :param return_dict: If True, return a dictionary instead of a Namespace.
    :return: A Namespace or dictionary with only relevant arguments.
    """
    # Get the argument specification of the target class constructor
    argspec = inspect.getfullargspec(target_cls.__init__)
    target_args = set(argspec.args)  # Ensure it's a set for fast lookup

    # Convert Namespace to dictionary if needed
    if isinstance(args, Namespace):
        args = vars(args)

    # Filter only arguments that exist in the target class
    filtered_args = {k: v for k, v in args.items() if k in target_args}

    # Return as dict or Namespace
    return filtered_args if return_dict else Namespace(**filtered_args)



def filter_function_args(args: Union[Namespace, dict], function, return_dict=False):
    argspec = inspect.getfullargspec(function)
    target_args = argspec.args
    if isinstance(args, Namespace):
        args = vars(args)
    filtered_args = {k: args[k] for k in target_args if k in args}
    if return_dict:
        return filtered_args
    return Namespace(**filtered_args)
