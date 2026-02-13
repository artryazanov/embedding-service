
from main import detect_model_profile


def test_detect_e5_standard():
    profile = detect_model_profile("intfloat/multilingual-e5-large")
    assert profile.max_seq_length == 512
    # Verify prefixing logic via format_text
    assert profile.format_text("hello", is_query=True) == "query: hello"
    assert profile.format_text("hello", is_query=False) == "passage: hello"


def test_detect_e5_custom():
    profile = detect_model_profile("my-custom-e5-v2")
    assert profile.max_seq_length == 512
    assert profile.format_text("test", is_query=True) == "query: test"


def test_detect_bge_m3_standard():
    profile = detect_model_profile("BAAI/bge-m3")
    assert profile.max_seq_length == 8192
    # BGE-M3 (Generic) does not add prefixes
    assert profile.format_text("hello", is_query=True) == "hello"
    assert profile.format_text("hello", is_query=False) == "hello"


def test_detect_bge_m3_custom():
    profile = detect_model_profile("custom/bge-m3-finetuned")
    assert profile.max_seq_length == 8192
    assert profile.format_text("hello") == "hello"


def test_detect_unsupported_model():
    # Unsupported models now default to GenericModelProfile instead of raising ValueError
    profile = detect_model_profile("bert-base-uncased")
    assert profile.max_seq_length == 512  # Default for generic
    assert profile.format_text("hello") == "hello"
