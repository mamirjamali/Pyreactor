class Tensor():
    """A strided tensor view over a flat storage sequence.

    Strides and offset are measured in storage elements. Integer indexing
    selects a row; tuple indexing selects an individual element.
    """

    def __init__(self, storage, shape, strides=None, offset=0):
        """Create a tensor over storage with the given shape and layout.

        If strides are omitted, a contiguous row-major layout is used.
        """

        self.storage = storage
        self.shape = tuple(shape)
        self.offset = offset


        if strides is None:
            self.strides = self._compute_contiguous_strides(self.shape)
        else:
            self.strides = tuple(strides)
        
        self._validate_metadata()

    def _compute_contiguous_strides(self, shape):
        """Return row-major strides for the given dimensions."""
        strides = [0] * len(shape)

        stride = 1

        for i in range(len(shape) - 1, -1, -1):
           strides[i] = stride
           stride *= shape[i]

        return tuple(strides)

    def _index(self, indices):
        """Translate a coordinate tuple to a storage element index."""
        self._validate_indices(indices)
        index = self.offset
        for i in range(0 , len(indices), 1):
            index += self.strides[i] * indices[i]

        return index
    
    def _sub(self, row):
        """Return a one-dimensional view of the selected row of a 2D tensor."""
        col = self.shape[-1]
        offset = self._index((row, 0))

        """Creating sub-tensor"""
        return Tensor(
            self.storage,
            (col,),
            (self.strides[-1],),
            offset)
    
    def transpose(self):
        shape = [self.shape[-1], self.shape[0]]
        strides = [self.strides[-1], self.strides[0]]


        return Tensor(self.storage, shape, strides, self.offset)

    def is_contiguous(self):
        return self.strides == self._compute_contiguous_strides(self.shape)

    def contiguous(self):
        if self.is_contiguous():
            return self
        else:
            storage = []
            rows = self.shape[0]
            cols = self.shape[1]

            for i in range(rows):
                for j in range(cols):
                    storage.append(self[i, j])

            return Tensor(storage, self.shape)

    def _validate_metadata(self):
        """Validate shape, strides, offset, and storage bounds."""

        # 1. Shape must contain non-negative integers.
        for dim in self.shape:
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise TypeError("Shape dimensions must be integers.")

            if dim < 0:
                raise ValueError("Shape dimensions cannot be negative.")

        # 2. There must be one stride per dimension.
        if len(self.strides) != len(self.shape):
            raise ValueError(
                "The number of strides must match the number of dimensions."
            )

        # 3. Negative strides are not supported yet.
        for stride in self.strides:
            if not isinstance(stride, int) or isinstance(stride, bool):
                raise TypeError("Strides must be integers.")

            if stride < 0:
                raise ValueError("Negative strides are not supported yet.")

        # 4. Validate the offset.
        if not isinstance(self.offset, int) or isinstance(self.offset, bool):
            raise TypeError("Offset must be an integer.")

        if self.offset < 0:
            raise ValueError("Offset cannot be negative.")

        if self.offset > len(self.storage):
            raise ValueError("Offset exceeds the storage boundary.")

        # 5. Empty tensors contain no elements to access.
        if any(dim == 0 for dim in self.shape):
            return

        # 6. Find the highest storage position this tensor can access.
        max_index = self.offset

        for dim, stride in zip(self.shape, self.strides):
            max_index += (dim - 1) * stride

        if max_index >= len(self.storage):
            raise ValueError(
                f"Tensor metadata exceeds storage bounds: "
                f"maximum index is {max_index}, "
                f"but storage length is {len(self.storage)}."
            )
    
    def _validate_indices(self, indices):
        """Validate the number and bounds of logical indices."""

        if len(indices) != len(self.shape):
            raise IndexError(
                f"Expected {len(self.shape)} indices, "
                f"but received {len(indices)}."
            )

        for axis, (index, dim) in enumerate(zip(indices, self.shape)):
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError(
                    f"Index for axis {axis} must be an integer."
                )

            if index < 0 or index >= dim:
                raise IndexError(
                    f"Index {index} is out of bounds for axis {axis} "
                    f"with size {dim}."
                )

    def __getitem__(self, indices):
        """Return a row view for an integer index or an element for a tuple."""
        if isinstance(indices, int):
            if len(self.shape) == 1:
                index = self._index((indices,))
                return self.storage[index]

            return self._sub(indices)

        index = self._index(indices)
        return self.storage[index]

    def __setitem__(self, indices, value):
        if isinstance(indices, int):
            indices = (indices,)

        index = self._index(indices)
        """Setting value at index"""
        self.storage[index] = value


t = Tensor([1, 2, 3, 4, 5, 6], (2, 3))

transposed = t.transpose()
print(transposed.storage)    # (3, 2)
print(transposed.shape)    # (3, 2)
print(transposed.strides)  # (1, 3)
print(transposed[0, 1])    # 4

contiguous_t = transposed.contiguous()

print(contiguous_t.storage)  # [1, 4, 2, 5, 3, 6]
print(contiguous_t.strides)  # (2, 1)
print(contiguous_t.is_contiguous())  # True