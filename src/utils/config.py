"""
Configuration loading and validation for the fall detection system.

Loads parameters from config/config.yaml and provides typed access
to all configurable values. No hardcoded constants in application code.

Reference: research/11.1 - Centralized configuration management.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """
    Loads and provides access to the YAML configuration file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        self._config = self._load(config_path)

    @staticmethod
    def _load(path: Path) -> Dict[str, Any]:
        """
        Load and parse the YAML configuration file.

        Parameters
        ----------
        path : Path
            Absolute path to the YAML file.

        Returns
        -------
        dict
            Parsed configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value using dot-separated key notation.

        Parameters
        ----------
        key : str
            Dot-separated key path, e.g. 'model.hidden_size'.
        default : Any
            Default value if key is not found.

        Returns
        -------
        Any
            The configuration value.
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    @property
    def seed(self) -> int:
        """Random seed for reproducibility."""
        return self._config.get("seed", 42)

    @property
    def raw(self) -> Dict[str, Any]:
        """Return the raw configuration dictionary."""
        return self._config
