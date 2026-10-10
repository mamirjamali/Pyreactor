class Tensor():

    def __init__(self, storage, shape, strides=None, offset=0):

        self.storage = storage
        self.shape = tuple(shape)
        self.offset = offset


        if strides is None:
            self.strides = self._compute_contiguous_strides(self.shape)
        else:
            self.strides = tuple(strides)

    def _compute_contiguous_strides(self, shape):
        strides = [0] * len(shape)

        stride = 1

        for i in range(len(shape) - 1, -1, -1):
           strides[i] = stride
           stride *= shape[i]

        return tuple(strides)

    def _index(self, indices):
        index = self.offset
        
        for i in range(0 , len(indices), 1):
            index += self.strides[i] * indices[i]

        return index
    
    def _sub(self, row):
        col = self.shape[-1]
        offset = self._index((2, 0))

        return Tensor(self.storage, (col,), (1,), offset)
        
    def __getitem__(self, indices):
        if isinstance(indices, int):
            return self._sub(indices)

        index = self._index(indices)
        return self.storage[index]