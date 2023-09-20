import time


class Timer:
    def __init__(self, label):
        self.label = label
        self.start_time = time.perf_counter()

    def stop(self):
        end_time = time.perf_counter()
        print(f"{self.label} completed in {end_time - self.start_time:0.4f} seconds")
