import selectors

from ai4sts2.kernel import Kernel, KernelError, engine


class Pool:
    def __init__(self, count, executable=None):
        self.executable = executable or engine()
        self.kernels = []
        try:
            for _ in range(count):
                self.kernels.append(Kernel(self.executable, connect=False))
            for kernel in self.kernels:
                kernel.connect()
        except BaseException:
            self.close()
            raise
        self.selector = selectors.DefaultSelector()
        self.pending = set()

    def __len__(self):
        return len(self.kernels)

    def send(self, worker, op, **arguments):
        if worker in self.pending:
            raise RuntimeError("A request is already pending for this kernel.")
        self.kernels[worker].send(op, **arguments)
        self.selector.register(self.kernels[worker], selectors.EVENT_READ, worker)
        self.pending.add(worker)

    def receive(self, timeout=None):
        results = []
        for key, _ in self.selector.select(timeout):
            worker = key.data
            self.selector.unregister(key.fileobj)
            self.pending.discard(worker)
            try:
                results.append((worker, self.kernels[worker].receive()))
            except KernelError as error:
                results.append((worker, error))
        return results

    def restart(self, worker):
        if worker in self.pending:
            self.selector.unregister(self.kernels[worker])
            self.pending.discard(worker)
        self.kernels[worker].close()
        self.kernels[worker] = Kernel(self.executable)

    def close(self):
        for kernel in self.kernels:
            kernel.close()
        self.kernels = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
