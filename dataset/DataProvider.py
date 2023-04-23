from dataset.StreamGenerator import StreamGenerator
from dataset.BatchGenerator import Batch
import math

class DataProvider:

    def generate_drift_stream(self, n_chunks, chunk_size, n_classes, n_drifts,
                              random_state=931604):
        args = {
            "n_classes": n_classes,
            "n_drifts": n_drifts,
            "n_chunks": n_chunks,
            "chunk_size": chunk_size,
            "random_state": random_state,
            "n_informative": 2,
            "n_redundant": 0,
            "n_repeated": 0,
            "n_features": 2,
            "recurring": False,
            "n_clusters_per_class": 2,
        }
        stream = StreamGenerator(**args)
        return stream
    
class BatchDivider():
    def __init__(self,
                 X,
                 y,
                 mini_batch_size=120, first_chunk_size=500):
        self.mini_batch_size = mini_batch_size
        self.max_iter = math.ceil(X.shape[0] / mini_batch_size)
        self.cur_iter = 0
        self.batch = Batch(X, y, chunk_size=mini_batch_size, first_chunk_size=first_chunk_size)
        self.cur_data = None
        self.next_data = self.batch.get_chunk()
    
    def next_task(self):
        if self.cur_iter >= self.max_iter-1:
            raise Exception('Number of tasks exceeded!')
        else:
            self.cur_data = self.next_data
            self.next_data = self.batch.get_chunk()
            self.cur_iter += 1
            if self.cur_data == None:
                self.cur_data = self.next_data
                self.next_data = self.batch.get_chunk()
                self.cur_iter += 1

            return self.cur_data[0], self.cur_data[1], self.next_data[0], self.next_data[1]