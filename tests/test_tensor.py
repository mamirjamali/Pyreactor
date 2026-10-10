from pyreactor.tensor import Tensor


def test_integer_index_returns_contiguous_row_view():
    t = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
    row = t[2]

    assert row.shape == (2,)
    assert row.strides == (1,)
    assert row.offset == 4
    assert row[0] == 5
    assert row[1] == 6
    assert row.storage is t.storage


def test_integer_index_returns_strided_row_view():
    t = Tensor([1, 2, 3, 4, 5, 6], (3, 2), strides=(1, 3))
    row = t[2]

    assert row.shape == (2,)
    assert row.strides == (3,)
    assert row.offset == 2
    assert row[0] == 3
    assert row[1] == 6
    assert row.storage is t.storage
