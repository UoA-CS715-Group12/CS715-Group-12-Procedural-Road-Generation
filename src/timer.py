import time


class Timer:
    def __init__(self, label):
        self.label = label
        self.start_time = time.perf_counter()

    def stop(self):
        end_time = time.perf_counter()
        time_taken = end_time - self.start_time
        print(f"{self.label} completed in {time_taken:0.4f} seconds")
        return time_taken
