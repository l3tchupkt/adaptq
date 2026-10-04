"""Security regression tests for the HuggingFace Transformers backend."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from adaptq.runtime_py.metadata import ModelConfig


def test_model_config_disables_remote_code_by_default():
    cfg = ModelConfig(model_path="example/model")
    assert cfg.allow_remote_code is False


def test_transformers_load_model_defaults_to_safe_remote_code_policy(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import adaptq.runtime_py.backends.transformers_hf as hf_backend

    tokenizer = SimpleNamespace(eos_token_id=2)
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path="example/model",
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=16,
            vocab_size=100,
            num_key_value_heads=4,
        ),
        eval=Mock(),
    )

    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setattr(hf_backend.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(hf_backend.AutoModelForCausalLM, "from_pretrained", model_loader)
    monkeypatch.setattr(hf_backend.torch.cuda, "is_available", lambda: False)

    adapter = hf_backend.TransformersAdapter()
    assert adapter.load_model(ModelConfig(model_path="example/model")) is True

    assert tokenizer_loader.call_args.kwargs["trust_remote_code"] is False
    assert model_loader.call_args.kwargs["trust_remote_code"] is False
    model.eval.assert_called_once_with()


def test_transformers_load_model_allows_explicit_remote_code_opt_in(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import adaptq.runtime_py.backends.transformers_hf as hf_backend

    tokenizer = SimpleNamespace(eos_token_id=2)
    model = SimpleNamespace(
        config=SimpleNamespace(
            _name_or_path="example/model",
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=16,
            vocab_size=100,
            num_key_value_heads=4,
        ),
        eval=Mock(),
    )

    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setattr(hf_backend.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(hf_backend.AutoModelForCausalLM, "from_pretrained", model_loader)
    monkeypatch.setattr(hf_backend.torch.cuda, "is_available", lambda: False)

    adapter = hf_backend.TransformersAdapter()
    cfg = ModelConfig(model_path="example/model", allow_remote_code=True)
    assert adapter.load_model(cfg) is True

    assert tokenizer_loader.call_args.kwargs["trust_remote_code"] is True
    assert model_loader.call_args.kwargs["trust_remote_code"] is True


def test_transformers_load_tokenizer_is_safe_without_model_config(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import adaptq.runtime_py.backends.transformers_hf as hf_backend

    tokenizer_loader = Mock(return_value=SimpleNamespace(eos_token_id=2))
    monkeypatch.setattr(hf_backend.AutoTokenizer, "from_pretrained", tokenizer_loader)

    adapter = hf_backend.TransformersAdapter()
    assert adapter.load_tokenizer("example/tokenizer") is True
    assert tokenizer_loader.call_args.kwargs["trust_remote_code"] is False
