import copy
import os.path as osp
import shutil

import yaml

from labelme.logger import logger

here = osp.dirname(osp.abspath(__file__))


def update_dict(target_dict, new_dict, validate_item=None):
    for key, value in new_dict.items():
        if validate_item:
            validate_item(key, value)
        if key not in target_dict:
            logger.warn("Skipping unexpected key in config: {}".format(key))
            continue
        if isinstance(target_dict[key], dict) and isinstance(value, dict):
            update_dict(target_dict[key], value, validate_item=validate_item)
        else:
            target_dict[key] = value


# -----------------------------------------------------------------------------


def get_default_config():
    config_file = osp.join(here, "default_config.yaml")
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # save default config to ~/.labelmerc
    user_config_file = osp.join(osp.expanduser("~"), ".labelmerc")
    if not osp.exists(user_config_file):
        try:
            shutil.copy(config_file, user_config_file)
        except Exception:
            logger.warn("Failed to save config: {}".format(user_config_file))

    return config


def validate_config_item(key, value):
    if key == "validate_label" and value not in [None, "exact"]:
        raise ValueError(
            "Unexpected value for config key 'validate_label': {}".format(value)
        )
    if key == "shape_color" and value not in [None, "auto", "manual"]:
        raise ValueError(
            "Unexpected value for config key 'shape_color': {}".format(value)
        )
    if key == "labels" and value is not None and len(value) != len(set(value)):
        raise ValueError(
            "Duplicates are detected for config key 'labels': {}".format(value)
        )


def get_user_config_path():
    """Return the default path for the user config file."""
    return osp.join(osp.expanduser("~"), ".labelmerc")


def save_shortcuts(config_path, shortcuts_overrides):
    """
    Save shortcut overrides to the user config file.
    shortcuts_overrides: dict of config_key -> value (only keys that differ from default).
    Merges with existing config, updating only the shortcuts section.
    Keys with value None are removed from user overrides (reset to default).
    """
    config_path = osp.expanduser(config_path)
    default_config = get_default_config()
    default_shortcuts = default_config.get("shortcuts", {})

    if osp.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warn("Failed to read config from {}: {}".format(config_path, e))
            config = {}
        if config is None or not isinstance(config, dict):
            config = {}
    else:
        config = {}

    # Use full default as base when file is empty; preserves structure
    if not config:
        config = copy.deepcopy(default_config)
    if "shortcuts" not in config:
        config["shortcuts"] = {}

    for key, value in shortcuts_overrides.items():
        if value is None or value == default_shortcuts.get(key):
            config["shortcuts"].pop(key, None)
        else:
            config["shortcuts"][key] = value

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _load_config_from_source(source):
    """Load config dict from a file path or YAML string. Returns None on failure."""
    if source is None:
        return None
    if isinstance(source, dict):
        return source
    s = str(source).strip()
    path = osp.expanduser(s)
    if "\n" not in s and osp.isfile(path):
        try:
            with open(path, "r") as f:
                logger.info("Loading config file from: {}".format(path))
                return yaml.safe_load(f)
        except Exception as e:
            logger.warn("Failed to load config from {}: {}".format(path, e))
            return None
    try:
        return yaml.safe_load(s)
    except Exception:
        return None


def get_config(config_file_or_yaml=None, config_from_args=None):
    # 1. default config
    config = get_default_config()

    # 2. specified as file or yaml
    if config_file_or_yaml is not None:
        config_from_yaml = _load_config_from_source(config_file_or_yaml)
        if isinstance(config_from_yaml, dict):
            update_dict(config, config_from_yaml, validate_item=validate_config_item)

    # 3. command line argument or specified config file
    if config_from_args is not None:
        update_dict(config, config_from_args, validate_item=validate_config_item)

    return config
