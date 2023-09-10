import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulate getting data in real-time
def get_next_data_point():
    return np.random.rand()*10

fig, ax = plt.subplots()
xdata, ydata = [0], [get_next_data_point()]
ln, = plt.plot([], [], 'r-')

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return ln,

def update(frame):
    xdata.append(xdata[-1] + 1)  # Increment the x-axis
    ydata.append(get_next_data_point())  # Get the new data point
    
    if len(xdata) > 10:  # Keep the last 10 data points
        del xdata[0]
        del ydata[0]

    ax.set_xlim(xdata[0], xdata[-1])  # Adjust the x-axis limits dynamically
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True, interval=1000)  # interval is in milliseconds
plt.show()
