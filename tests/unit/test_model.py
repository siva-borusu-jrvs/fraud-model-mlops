"""Dummy unit tests — always pass. Used to validate CI pipeline."""


def test_addition():
    assert 1 + 1 == 2


def test_string_concatenation():
    assert "hello" + " " + "world" == "hello world"


def test_list_length():
    assert len([1, 2, 3]) == 3


def test_dict_key_access():
    d = {"a": 1, "b": 2}
    assert d["a"] == 1
