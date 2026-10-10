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
        index = self.offset
        
        for i in range(0 , len(indices), 1):
            index += self.strides[i] * indices[i]

        return index
    
    def _sub(self, row):
        """Return a one-dimensional view of the selected row of a 2D tensor."""
        col = self.shape[-1]
        offset = self._index((row, 0))

        return Tensor(
            self.storage,
            (col,),
            (self.strides[-1],),
             offset)
        
    def __getitem__(self, indices):
        """Return a row view for an integer index or an element for a tuple."""
        if isinstance(indices, int):
            return self._sub(indices)

        index = self._index(indices)
        return self.storage[index]