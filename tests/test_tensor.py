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

def test_integer_index_set_storage():
    t = Tensor([1, 2, 3, 4, 5, 6], (3, 2), strides=(2, 1))
    row = t[2]
    row[0] = 100

    assert t[2,0] == row[0] == 100
    assert t.storage[4] == row.storage[4] == 100
    assert t.storage is row.storage

def test_transposed_2d_array():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    transposed = t.transpose()

    assert transposed.shape == (3, 2)
    assert transposed.strides == (1, 3)
    assert transposed.offset == t.offset
    assert transposed.storage is t.storage

    assert transposed[0, 1] == 4
    assert transposed[2, 1] == 6

    transposed[0, 1] = 99
    assert t[1, 0] == 99

def test_contiguous_transition():
    t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
    transposed = t.transpose()
    contiguous_t = transposed.contiguous()

    assert contiguous_t.storage != t.storage
    assert contiguous_t.storage[1] == t.storage[3]