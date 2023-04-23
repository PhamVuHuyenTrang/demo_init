import numpy as np
from scipy.stats import logistic
from sklearn.datasets import make_classification
import csv
import math


class Batch:

    def __init__(self, X, y, n_chunks=None, chunk_size=None, first_chunk_size= None):
        assert (n_chunks is None or chunk_size is None)
        assert (n_chunks is not None or chunk_size is not None)
        if n_chunks is not None:
            self.chunk_size = int(X.shape[0] - first_chunk_size / (n_chunks-1))
            self.n_chunks = n_chunks
        else:
            self.chunk_size = chunk_size
            self.n_chunks = math.ceil(X.shape[0] - first_chunk_size / chunk_size) + 1
        self.X = X
        self.y = y
        self.current_chunk = None
        self.chunk_id = -1
        self.previous_chunk = None
        self.first_chunk_size = first_chunk_size

    def is_dry(self):
        return (self.chunk_id + 1 >= self.n_chunks if hasattr(self, "chunk_id") else False)

    def get_chunk(self):
        if hasattr(self, "X"):
            self.previous_chunk = self.current_chunk

        self.chunk_id += 1
        if self.chunk_id == 0:
            start, end = (0, self.first_chunk_size)
            print("Start")
            print(start)
            print("End")
            print(end)
            self.current_chunk = (self.X[start:end], self.y[start:end])
            return self.current_chunk
        else:
            if self.chunk_id < self.n_chunks:
                start, end = (
                    self.chunk_size * (self.chunk_id-1) + self.first_chunk_size,
                    self.chunk_size * (self.chunk_id-1) + self.first_chunk_size + self.chunk_size,
                )
                if end < self.X.shape[0]:
                    self.current_chunk = (self.X[start:end], self.y[start:end])
                else:
                    self.current_chunk = (self.X[start:self.X.shape[0]], self.y[start:self.X.shape[0]])
                return self.current_chunk
            else:
                return None

if __name__ == "__main__":
    x = np.array(range(102)).reshape(51, 2)
    y = np.array(range(51))
    batch = Batch(x, y, chunk_size=10)
    while not batch.is_dry():
        xy = batch.get_chunk()
        x = xy[0]
        y = xy[1]
        print(f'x: {x} --- y: {y}')
