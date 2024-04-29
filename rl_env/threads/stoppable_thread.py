from threading import Thread, Event
from time import sleep


class StoppableThread(Thread):
    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
    
    def run(self):
        while not self.stopped():
            self._target(*self._args, **self._kwargs)