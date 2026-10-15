from collections.abc import Iterable, Iterator


def iter_coordinates(shape: Iterable[int]) -> Iterator[tuple[int, ...]]:
    """Yield every valid coordinate for a shape in row-major order.

    Args:
        shape: An iterable of non-negative integer dimensions.

    Yields:
        Coordinate tuples in row-major order.

    Raises:
        TypeError: If the shape or any dimension has an invalid type.
        ValueError: If any dimension is negative.

    Examples:
        >>> list(iter_coordinates((2, 2)))
        [(0, 0), (0, 1), (1, 0), (1, 1)]

        >>> list(iter_coordinates((2, 1)))
        [(0, 0), (1, 0)]

        >>> list(iter_coordinates((0, 3)))
        []

        >>> list(iter_coordinates(()))
        [()]
    """
    try:
        dimensions = tuple(shape)
    except TypeError as exc:
        raise TypeError("Shape must be an iterable of integers.") from exc

    for dim in dimensions:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("Shape dimensions must be integers.")
        if dim < 0:
            raise ValueError("Shape dimensions cannot be negative.")

    # A scalar has one coordinate: the empty tuple.
    if not dimensions:
        yield ()
        return

    # If any dimension is zero, there are no valid coordinates.
    if any(dim == 0 for dim in dimensions):
        return

    indices = [0] * len(dimensions)

    while True:
        yield tuple(indices)

        # Increment the rightmost axis, carrying when necessary.
        axis = len(dimensions) - 1

        while axis >= 0:
            indices[axis] += 1

            if indices[axis] < dimensions[axis]:
                break

            indices[axis] = 0
            axis -= 1

        # Carry passed the leftmost axis: iteration is complete.
        if axis < 0:
            return