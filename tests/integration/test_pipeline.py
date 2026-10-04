"""Dummy integration tests — always pass. Used to validate CI pipeline."""


def test_tuple_unpacking():
    a, b = 1, 2
    assert a + b == 3


def test_string_upper():
    assert "hello".upper() == "HELLO"


def test_list_sort():
    items = [3, 1, 2]
    assert sorted(items) == [1, 2, 3]
