from zipfft import foo
from zipfft import binding


def test_foo():
    """Test the foo function."""
    assert foo(1) == 2
    assert foo(0) == 1
    assert foo(-1) == 0
    assert foo(100) == 101


def test_binding():
    result = binding.add(1, 2)
    assert result == 3
