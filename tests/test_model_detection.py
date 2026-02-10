import pytest

from main import ModelProfile, detect_model_profile


def test_detect_e5_standard():
    profile = detect_model_profile("intfloat/multilingual-e5-large")
    assert profile.requires_prefix is True
    assert profile.max_seq_length == 512
    assert profile.query_prefix == "query: "
    assert profile.passage_prefix == "passage: "


def test_detect_e5_custom():
    profile = detect_model_profile("my-custom-e5-v2")
    assert profile.requires_prefix is True
    assert profile.max_seq_length == 512


def test_detect_bge_m3_standard():
    profile = detect_model_profile("BAAI/bge-m3")
    assert profile.requires_prefix is False
    assert profile.max_seq_length == 8192
    assert profile.query_prefix == ""
    assert profile.passage_prefix == ""


def test_detect_bge_m3_custom():
    profile = detect_model_profile("custom/bge-m3-finetuned")
    assert profile.requires_prefix is False
    assert profile.max_seq_length == 8192


def test_detect_unsupported_model():
    with pytest.raises(ValueError) as excinfo:
        detect_model_profile("bert-base-uncased")
    assert "Unsupported model architecture" in str(excinfo.value)
