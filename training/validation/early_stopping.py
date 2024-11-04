class EarlyStopping:
    def __init__(
            self,
            patience: int,
            min_delta: float,
            monitor: str
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_value = float('inf')
        self.should_stop = False

    def __call__(self, current_value: float) -> bool:
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience
        return self.should_stop
