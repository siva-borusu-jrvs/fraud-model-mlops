"""Dummy unit tests — always pass. Used to validate CI pipeline."""


def test_boolean_true():
    assert True


def test_set_membership():
    assert "a" in {"a", "b", "c"}


def test_integer_comparison():
    assert 10 > 5


def test_none_is_falsy():
    assert not None
