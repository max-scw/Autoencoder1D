import re
import logging
from ast import literal_eval

from typing import Union, List, Dict, Tuple


re_number = re.compile("^[0-9.,]+$")
re_integer = re.compile(r"^\d+$")
re_float = re.compile(r"^((\d+\.(\d+)?)|(\.\d+))$")
re_float_de = re.compile(r"^((\d+,(\d+)?)|(,\d+))$")
re_boolean = re.compile(r"^(true|false)$", re.IGNORECASE | re.ASCII)
re_list_or_tuple_or_dict = re.compile(r"^\s*(\[.*\]|\(.*\)|\{.*\})\s*$", re.ASCII)
re_comma = re.compile(r"^(\".*\")|(\'.*\')$", re.ASCII)


def cast(var: str) -> Union[None, int, float, str, bool]:
    """casting strings to primitive datatypes"""
    if re_number.match(var):
        if re_integer.match(var):  # integer
            var = int(var)
        elif re_float.match(var):  # float
            var = float(var)
        elif re_float_de.match(var):  # float
            var = float(var.replace(",", "."))
    elif re_boolean.match(var):
        var = True if var[0].lower() == "t" else False
    elif re_list_or_tuple_or_dict.match(var):
        var = literal_eval(var)
    elif re_comma.match(var):
        # strip enclosing high comma
        var = var.strip('"').strip('"')
    return var


def cast_logging_level(var: str, default: int = logging.INFO) -> int:
    """Only casts logging levels"""
    # cast string if possible
    if isinstance(var, str):
        var = cast(var)

    options = {
        "debug": logging.DEBUG,  # 10
        "info": logging.INFO,  # 20
        "warning": logging.WARNING,  # 30
        "warn": logging.WARN,  # 30
        "error": logging.ERROR,  # 40
        "critical": logging.CRITICAL,  # 50
        "fatal": logging.FATAL,  # 50
        "notset": logging.NOTSET  # 0
    }
    if isinstance(var, int):
        if var not in options.values():
            return default

    elif isinstance(var, str):
        for ky, val in options.items():
            if var.lower() == ky:
                return val
    else:
        return default
