import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax1 = self.ax.twinx()

        # Initialize plot elements
        self.line,  = self.ax.plot([], [], lw=2)
        self.point, = self.ax.plot([], [], 'ro')

    def plot(self, x_data, y_data, label=None, color=None, linestyle=None):
        self.ax.plot(x_data, y_data, label=label, color=color, linestyle=linestyle)

    def plotsinvtrapped(self, x_data, trapped_data, sin_data):
        color = 'tab:red'
        self.ax.set_xlabel('t(s)')
        self.ax.set_ylabel('Trapped', color=color)
        self.ax.plot(x_data, trapped_data, color=color)
        self.ax.tick_params(axis='y', labelcolor=color)

        # Creating a second y-axis and plotting the second dataset
        color = 'tab:blue'
        self.ax1.set_ylabel('∇²E(V)', color=color)
        plt.text(max(x_data) + 0.1*(max(x_data)-min(x_data)), max(sin_data), '× 2.5*10^10', color=color, horizontalalignment='left', verticalalignment='center')

        self.ax1.plot(x_data, sin_data, color=color)
        self.ax1.tick_params(axis='y', labelcolor=color)

        # Adding title
        plt.title('Plot trapping particles at a frequency of 500 Hertz')

    def addlabel(self, xlabel, ylabel):
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def addlabel(self, xlabel, y1label, y2label):
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(y1label)
        self.ax1.set_ylabel(y2label)

    def scatter(self, x_data, y_data, label=None, color=None):
        self.ax.scatter(x_data, y_data, label=label, color=color)

    def show(self):
        self.ax.legend()
        plt.show()

    # Initialization function
    def init(self):
        self.line.set_data([], [])
        self.point.set_data([], [])
        return self.line, self.point

    # Function to update the plot
    def update(self, frame, time, data, line, point):
        line.set_data(time, data)
        point.set_data(time[frame], data[frame])
        return line, point

    def animate(self, time, data):
        ani = animation.FuncAnimation(self.fig, self.update, np.shape(time)[0]-1, fargs=(time, data, self.line, self.point), init_func=self.init,  interval=1, blit=True)
        ani.save('sin.gif', fps=100)
        return

    def save(self, path, name):
        plt.savefig(str(path) + '/' + name)
        return


class Plotter3D:
    def __init__(self, position_data):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.position = position_data
        x_history, y_history, z_history = position_data[0, :], position_data[1, :], position_data[2, :]
        self.line, = self.ax.plot(x_history, y_history, z_history, 'r-')

    def plot3d(self):
        self.scatter(self.position[0], 'g-', 'Initial Position')
        self.scatter([0, 0, 0], 'r-', 'Trap Position')
        self.ax.legend()

    # Function to compute color based on y-values
    # def compute_color(y):
    #     return 'b' if a_sin[y] > 1 else 'r'

    # Function to update the plot
    def update(self, frame, data, line):
        line.set_data(data[:2, :frame])
        line.set_3d_properties(data[2, :frame])
        #line.set_color(compute_color(frame))  # Color based on the first y-value
        return line

    def animate3d(self, path):
        self.plot3d()
        ani = animation.FuncAnimation(self.fig, self.update, np.shape(self.position)[1], fargs=(self.position, self.line),  interval=1)
        ani.save(str(path) + '/' + 'simulation.gif', fps=100)

    def scatter(self, position3d, colour, label):
        x, y, z = position3d[0], position3d[1], position3d[2]
        self.ax.scatter(x, y, z, colour, marker='o', label=label)

    def save(self, path, name):
        plt.savefig(str(path) + '/' + name)
        return


