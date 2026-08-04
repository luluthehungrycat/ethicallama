"""Unit tests for :mod:`ethllama.profiles`.

The :class:`Profile` dataclass and its helpers (``list_profiles``,
``load_profile``, ``delete_profile``, ``profile_exists``) round-trip
profile YAML files in a user-controlled directory.  These tests
exercise that contract using ``tmp_path`` for isolation — no real
``~/.ethllama/profiles/`` is ever touched.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterator

import pytest
import yaml

from ethllama import profiles as profiles_mod
from ethllama.profiles import (
    PROFILES_DIR,
    Profile,
    delete_profile,
    list_profiles,
    load_profile,
    profile_exists,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profiles_dir(tmp_path, monkeypatch) -> Path:
    """Redirect ``PROFILES_DIR`` to a fresh tmp directory for each test."""
    target = tmp_path / "profiles"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(profiles_mod, "PROFILES_DIR", target)
    return target


@pytest.fixture
def make_profile(profiles_dir) -> Iterator:
    """Factory that writes a profile YAML to disk and returns the Profile.

    Usage::

        prof = make_profile("chat-py", model="qwen.gguf", temperature=0.3)
    """

    def _make(name: str, **fields) -> Profile:
        model = fields.pop("model", "/path/to/model.gguf")
        parameters = fields.pop("parameters", None)
        description = fields.pop("description", "")
        system_prompt = fields.pop("system_prompt", "")
        template = fields.pop("template", "")
        stop = fields.pop("stop", [])
        metadata = fields.pop("metadata", {})
        if parameters is None:
            parameters = fields
        prof = Profile(
            name=name,
            model=model,
            description=description,
            parameters=parameters,
            system_prompt=system_prompt,
            template=template,
            stop=stop,
            metadata=metadata,
        )
        prof.save(profiles_dir=profiles_dir)
        return prof

    return _make


# ---------------------------------------------------------------------------
# Profile dataclass: round trip and field defaults
# ---------------------------------------------------------------------------


def test_profile_default_field_values():
    """Profile() with no extra args has empty defaults, not None."""
    p = Profile(name="x", model="/m.gguf")
    assert p.parameters == {}
    assert p.system_prompt == ""
    assert p.template == ""
    assert p.stop == []
    assert p.metadata == {}
    assert p.description == ""


def test_profile_round_trip(make_profile, profiles_dir):
    """Save a profile, then read it back; every field is preserved."""
    original = Profile(
        name="chat-python",
        description="Default Python coding profile",
        model="Qwen3.5-0.8B-UD-IQ2_XXS",
        parameters={
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 30,
            "max_tokens": 2048,
            "n_gpu_layers": -1,
            "ctx_size": 4096,
        },
        system_prompt="You are an expert Python developer.",
        template="<|im_start|>system\n{{ .System }}<|im_end|>",
        stop=["<|im_end|>", "<|im_start|>"],
        metadata={"author": "tester", "version": 1},
    )
    path = original.save(profiles_dir=profiles_dir)
    assert path.exists()
    assert path.name == "chat-python.yaml"

    loaded = Profile.from_yaml(path)
    assert loaded == original
    # And explicitly verify each field, to catch subtle regressions
    # (e.g. accidental sharing of mutable defaults across instances).
    assert loaded.name == original.name
    assert loaded.model == original.model
    assert loaded.description == original.description
    assert loaded.parameters == original.parameters
    assert loaded.system_prompt == original.system_prompt
    assert loaded.template == original.template
    assert loaded.stop == original.stop
    assert loaded.metadata == original.metadata


def test_profile_to_yaml_format(profiles_dir):
    """to_yaml() produces a YAML string with the right top-level keys."""
    p = Profile(
        name="demo",
        model="/m.gguf",
        description="d",
        parameters={"temperature": 0.5},
        system_prompt="hi",
        template="t",
        stop=["</s>"],
        metadata={"k": "v"},
    )
    rendered = p.to_yaml()
    data = yaml.safe_load(rendered)
    assert data["name"] == "demo"
    assert data["model"] == "/m.gguf"
    assert data["description"] == "d"
    assert data["parameters"] == {"temperature": 0.5}
    assert data["system_prompt"] == "hi"
    assert data["template"] == "t"
    assert data["stop"] == ["</s>"]
    assert data["metadata"] == {"k": "v"}


def test_profile_from_yaml_uses_filename_when_name_missing(tmp_path, profiles_dir):
    """When the YAML has no 'name' key, the file stem is used."""
    yaml_path = profiles_dir / "fallback-name.yaml"
    yaml_path.write_text(
        "model: /some/model.gguf\n"
        "parameters:\n"
        "  temperature: 0.4\n",
        encoding="utf-8",
    )
    prof = Profile.from_yaml(yaml_path)
    assert prof.name == "fallback-name"
    assert prof.model == "/some/model.gguf"
    assert prof.parameters == {"temperature": 0.4}


def test_profile_from_yaml_requires_model(tmp_path, profiles_dir):
    """A YAML file without a 'model' key raises ValueError."""
    yaml_path = profiles_dir / "no-model.yaml"
    yaml_path.write_text("name: bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model"):
        Profile.from_yaml(yaml_path)


def test_profile_from_yaml_rejects_non_mapping(tmp_path, profiles_dir):
    """A YAML file that parses to a non-dict (e.g. a list) raises ValueError."""
    yaml_path = profiles_dir / "list.yaml"
    yaml_path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        Profile.from_yaml(yaml_path)


def test_profile_from_yaml_handles_optional_missing_fields(profiles_dir):
    """Optional fields absent from YAML default to empty values."""
    yaml_path = profiles_dir / "minimal.yaml"
    yaml_path.write_text("model: /m.gguf\n", encoding="utf-8")
    prof = Profile.from_yaml(yaml_path)
    assert prof.description == ""
    assert prof.parameters == {}
    assert prof.system_prompt == ""
    assert prof.template == ""
    assert prof.stop == []
    assert prof.metadata == {}


def test_profile_get_param_returns_default_when_missing():
    """get_param returns the provided default for missing keys."""
    p = Profile(name="x", model="/m.gguf", parameters={"temperature": 0.5})
    assert p.get_param("temperature") == 0.5
    assert p.get_param("top_k") is None
    assert p.get_param("top_k", default=40) == 40


def test_profile_save_creates_directory(tmp_path):
    """save() creates the target directory if it does not exist."""
    fresh = tmp_path / "new" / "profiles"
    assert not fresh.exists()
    prof = Profile(name="x", model="/m.gguf")
    out = prof.save(profiles_dir=fresh)
    assert out.exists()
    assert fresh.exists()


def test_profile_save_writes_valid_yaml(profiles_dir):
    """The written file is valid YAML that round-trips back into a Profile."""
    p = Profile(
        name="round-trip",
        model="/m.gguf",
        parameters={"temperature": 0.7, "top_p": 0.95},
    )
    out_path = p.save(profiles_dir=profiles_dir)
    with open(out_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["name"] == "round-trip"
    assert data["model"] == "/m.gguf"
    assert data["parameters"] == {"temperature": 0.7, "top_p": 0.95}


# ---------------------------------------------------------------------------
# list_profiles
# ---------------------------------------------------------------------------


def test_list_profiles_empty_when_dir_missing(tmp_path, monkeypatch):
    """list_profiles returns [] when the directory doesn't exist."""
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr(profiles_mod, "PROFILES_DIR", missing)
    assert list_profiles() == []


def test_list_profiles_empty_when_dir_empty(profiles_dir):
    """list_profiles returns [] when the directory has no YAMLs."""
    assert list_profiles() == []


def test_list_profiles_returns_sorted_names(make_profile):
    """list_profiles returns sorted profile names (no .yaml)."""
    make_profile("zebra")
    make_profile("alpha")
    make_profile("middle")

    names = list_profiles()
    assert names == ["alpha", "middle", "zebra"]


def test_list_profiles_ignores_non_yaml(make_profile, profiles_dir):
    """Files in the profiles dir without .yaml extension are ignored."""
    make_profile("good")
    (profiles_dir / "notes.txt").write_text("ignore me")
    (profiles_dir / "README.md").write_text("readme")

    assert list_profiles() == ["good"]


# ---------------------------------------------------------------------------
# load_profile
# ---------------------------------------------------------------------------


def test_load_profile_returns_profile(make_profile):
    """load_profile reads the YAML and returns a Profile dataclass."""
    saved = make_profile("chat", model="/m.gguf", temperature=0.42)
    loaded = load_profile("chat")
    assert isinstance(loaded, Profile)
    assert loaded.name == saved.name
    assert loaded.model == saved.model
    assert loaded.parameters == {"temperature": 0.42}


def test_load_profile_missing_raises(profiles_dir):
    """load_profile raises FileNotFoundError for unknown names."""
    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        load_profile("does-not-exist")


def test_load_profile_explicit_dir_arg(make_profile, profiles_dir):
    """An explicit profiles_dir arg bypasses PROFILES_DIR."""
    make_profile("alpha", model="/m.gguf")
    # Use a sub-directory as the "real" home; profiles_dir is the tmp.
    loaded = load_profile("alpha", profiles_dir=profiles_dir)
    assert loaded.name == "alpha"


# ---------------------------------------------------------------------------
# profile_exists
# ---------------------------------------------------------------------------


def test_profile_exists_true_when_present(make_profile):
    """profile_exists returns True for a profile that was saved."""
    make_profile("here")
    assert profile_exists("here") is True


def test_profile_exists_false_when_absent(profiles_dir):
    """profile_exists returns False for an unknown name."""
    assert profile_exists("never-created") is False


# ---------------------------------------------------------------------------
# delete_profile
# ---------------------------------------------------------------------------


def test_delete_profile_removes_file(make_profile, profiles_dir):
    """delete_profile unlinks the YAML and returns True."""
    path = make_profile("bye").save(profiles_dir=profiles_dir)
    assert path.exists()
    assert delete_profile("bye") is True
    assert not path.exists()


def test_delete_profile_missing_returns_false(profiles_dir):
    """delete_profile returns False when there is nothing to delete."""
    assert delete_profile("ghost") is False


def test_delete_profile_does_not_touch_other_profiles(make_profile, profiles_dir):
    """Deleting one profile does not affect siblings."""
    make_profile("a")
    make_profile("b")
    make_profile("c")
    assert delete_profile("b") is True
    remaining = list_profiles()
    assert remaining == ["a", "c"]


# ---------------------------------------------------------------------------
# Module-level PROFILES_DIR
# ---------------------------------------------------------------------------


def test_profiles_dir_default_points_to_ethllama_home():
    """The default PROFILES_DIR is ~/.ethllama/profiles/."""
    # We can't assert the exact path (varies by environment), but the
    # path should be a Path whose last component is "profiles" and
    # whose parent ends with ".ethllama".
    assert PROFILES_DIR.name == "profiles"
    assert PROFILES_DIR.parent.name == ".ethllama"
