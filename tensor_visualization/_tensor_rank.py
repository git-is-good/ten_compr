# an observable
class TensorData(object):
    def __init__(self):
        self.reset()
        self.observers = []

    def reset(self):
        self.tensor = None
        self.partitions = (2, 2, 2)
        self.ranks = []

    #TODO:
    def rank(i, j, k):
        return 100 if i == j and j == k else 0

    def setTensor(self, tensor):
        self.tensor = tensor

    def register(self, a_observer):
        self.observers.append(a_observer)

    # ValueError only if program error
    def unregister(self, a_observer):
        self.observers.remove(a_observer)

    def update(self):
        self._update_ranks()
        for ob in self.observers:
            ob.on_tensor_data_update(self)

    def _update_ranks(self):
        '''
        for given tensor, partitions, compute all involed ranks
        '''
        ...

