"""Tests for src.utils — config loading and path resolution."""

import pytest
import yaml

from src.utils import ROOT, load_config


class TestLoadConfig:
    """Tests for load_config()."""

    def test_default_path_returns_dict(self):
        """Loading with default path returns a valid config dict."""
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "dataset" in cfg
        assert "labels" in cfg
        assert "splits" in cfg

    def test_custom_path(self, tmp_path):
        """Loading from a custom YAML path works."""
        custom = {"foo": "bar", "nested": {"key": 42}}
        path = tmp_path / "custom.yaml"
        path.write_text(yaml.dump(custom))

        result = load_config(str(path))
        assert result == custom

    def test_missing_file_raises(self, tmp_path):
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))


class TestRoot:
    """Tests for the ROOT path constant."""

    def test_root_is_project_dir(self):
        """ROOT should point to the project root (parent of src/)."""
        assert (ROOT / "src").is_dir()
        assert (ROOT / "configs").is_dir()
