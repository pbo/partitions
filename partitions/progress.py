# coding: utf-8
import time
import datetime


class Progress(object):

    def __init__(self, indication_rate=1):
        self._indication_rate = indication_rate

    @property
    def current_progress(self):
        return self._progress / self._total

    def start(self, total=None):
        self._start_time = time.time()
        self._prev_indication_time = self._start_time
        self._progress = 0
        self._total = total
        self.print_current_progress()

    def progress(self, iterations=1, new_total=None):
        self._progress += iterations

        self._total = new_total if new_total is not None else self._total
        if self._total is None:
            return

        current_time = time.time()
        sec_since_prev_indication = current_time - self._prev_indication_time
        if sec_since_prev_indication >= self._indication_rate:
            self._prev_indication_time = current_time
            self.print_current_progress()

    def print_current_progress(self):
        current_time = time.time()
        elapsed_seconds = current_time - self._start_time

        if elapsed_seconds == 0 or self._progress == 0:
            print("{:.2%}    Elaspsed: {}    Remaining: {}".format(
                0, "00:00:00", "Unknown"))
        elif self._total is None:
            elapsed_str = str(datetime.timedelta(seconds=elapsed_seconds))
            print("??.??%    Elaspsed: {}    Remaining: {}".format(
                elapsed_str, "Unknown"))
        else:
            elapsed_str = str(datetime.timedelta(seconds=elapsed_seconds))
            speed = self._progress / elapsed_seconds
            remaining_progress = self._total - self._progress
            remaining_seconds = remaining_progress / speed
            if remaining_seconds < 0:
                remaining_seconds = 0
            remaining_str = str(datetime.timedelta(seconds=remaining_seconds))

            print("{:.2%}    Elaspsed: {}    Remaining: {}".format(
                self.current_progress,
                elapsed_str,
                remaining_str))
