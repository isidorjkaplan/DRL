from tensorboardX import SummaryWriter

class Plotter():
    def __init__(self, folder):
        self.writer = SummaryWriter(folder)
        self.last_written = {}

    def plot(self, tag, name, x, y):
        key = tag + '/' + name
        if key in self.last_written.keys() and x <= self.last_written[key]:
            #Cannot write if it has already been written!
            return
        self.last_written[key] = x
        self.writer.add_scalar(key, y, x)

    def close(self):
        self.writer.close()
        print("Closed plotter!")
            