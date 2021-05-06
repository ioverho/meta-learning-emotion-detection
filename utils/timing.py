import time

class Timer():
    def __init__(self):

        self.t0 = time.time()

    def dt(self):
        t1 = time.time()
        dt = t1 - self.t0

        m, s = divmod(dt, 60)
        h, m = divmod(m, 60)

        dt_print = '{:02.0f}:{:02.0f}:{:05.2f}'.format(h, m, s)

        return dt_print
