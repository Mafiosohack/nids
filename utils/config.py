"""
Configuration Management Utility
Loads and manages system configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager for NIDS system."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if not self._config:
            if config_path is None:
                project_root = Path(__file__).parent.parent
                config_path = project_root / "config" / "config.yaml"
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            print(f"INFO: Configuration loaded from {config_path}")
        except FileNotFoundError:
            print(f"ERROR: Configuration file not found: {config_path}")
            self._config = {}
        except yaml.YAMLError as e:
            print(f"ERROR: Error parsing YAML configuration: {e}")
            self._config = {}

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        keys = key_path.split('.')
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        return self._config.copy()


# Global configuration instance
config = Config()


def get_config() -> Config:
    return config


def get(key_path: str, default: Any = None) -> Any:
    return config.get(key_path, default)